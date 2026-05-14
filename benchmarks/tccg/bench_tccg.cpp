#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "tcont.hpp"
#include "tccg_cases.hpp"
#include "tccg_utils.hpp"
#include "tccg_verify.hpp"

int main(int argc, char *argv[])
{
    try
    {
        int target_tccg_benchmark = -1;
        bool verify = false;
        const std::vector<TconT::TCEquation>* benchmark_cases = &list_tccg_bench;
        
        for(int i = 1; i < argc; ++i) 
        {
            const char* arg = argv[i];

            if(std::strcmp(arg, "-b") == 0 || std::strcmp(arg, "--benchmark") == 0) 
            {
                if (i + 1 < argc) {
                    target_tccg_benchmark = atoi(argv[++i]);
                }
            } 
            else if(std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--help") == 0) 
            {
                printf("Usage: %s [-b <benchmark_id>] [--tf32|--fp32|--fp64] [--verify]\n", argv[0]);
                printf("  -b, --benchmark <id>       Specify TCCG benchmark ID (0-48)\n");
                printf("      --tf32                 Run the TF32 benchmark cases\n");
                printf("      --fp32                 Alias for --tf32 (backward compatibility)\n");
                printf("      --fp64                 Run the FP64 benchmark cases (default)\n");
                printf("      --verify               Run correctness verification after timing\n");
                return 0;
            }
            else if (std::strcmp(arg, "--tf32") == 0 || std::strcmp(arg, "--fp32") == 0)
            {
                benchmark_cases = &list_tccg_bench_fp32;
            }
            else if (std::strcmp(arg, "--fp64") == 0)
            {
                benchmark_cases = &list_tccg_bench;
            }
            else if (std::strcmp(arg, "--verify") == 0)
            {
                verify = true;
            }
        }

        if(target_tccg_benchmark < 0 || target_tccg_benchmark >= benchmark_cases->size()) {
            printf("Invalid benchmark ID: %d. Valid range is 0 to %lu.\n", target_tccg_benchmark, benchmark_cases->size() - 1);
            return -1;
        }

        const TconT::TCEquation& desc = (*benchmark_cases)[target_tccg_benchmark];
        const int64_t operation_count = 2 * static_cast<int64_t>(
            std::sqrt(
                static_cast<double>(compute_tensor_size(desc.modeA, desc.extent)) *
                static_cast<double>(compute_tensor_size(desc.modeB, desc.extent)) *
                static_cast<double>(compute_tensor_size(desc.modeC, desc.extent))));

        printf("-----------------------------------------------------------------------\n");
        printf("[TconT] Target TCCG's Benchmark: %d\n", target_tccg_benchmark);
        printf("[TconT] column-major format.\n");
        
        print_equation(desc);

        int64_t size_A = compute_tensor_size(desc.modeA, desc.extent);
        int64_t size_B = compute_tensor_size(desc.modeB, desc.extent);
        int64_t size_C = compute_tensor_size(desc.modeC, desc.extent);

        const size_t scalar_size = TconT::scalar_type_size(desc.scalar_type);
        printf("[TconT] scalar type: %s\n", TconT::scalar_type_name(desc.scalar_type));
        printf("[TconT]  Size A: %12ld (words), %16ld (bytes)\n", size_A, size_A * scalar_size);
        printf("[TconT]  Size B: %12ld (words), %16ld (bytes)\n", size_B, size_B * scalar_size);
        printf("[TconT]  Size C: %12ld (words), %16ld (bytes)\n", size_C, size_C * scalar_size);
        printf("[TconT] # FLOPS: %20ld\n", operation_count);
        printf("-----------------------------------------------------------------------\n");

        printf("[TconT] Selct TTGT or Direct\n");
        
        TconT::ExecutionPlan plan = TconT::plan(desc);
        TconT::ExecutionRun run = TconT::prepare(plan);
        TconT::RunOptions options;
        options.warmup = 100;
        options.repeats = 200;

        TconT::ContractionResult result = TconT::benchmark(run, desc, options);
        
        printf("[TconT] Avg Time: %10.4f ms\n", result.avg_ms);
        printf("[TconT] GFLOPS  : %10.4f GFLOPS\n", result.gflops);

        std::string validation = "SKIP";
        if (verify) {
            const size_t output_elements = static_cast<size_t>(compute_tensor_size(desc.modeC, desc.extent));
            const size_t output_bytes = output_elements * TconT::scalar_type_size(desc.scalar_type);
            std::vector<unsigned char> output_storage(output_bytes);

            TconT::copy_output_to_host(run, output_storage.data(), output_bytes);
            const VerificationResult verification_result = verify_tccg_case(
                static_cast<size_t>(target_tccg_benchmark),
                desc,
                output_storage.data(),
                TconT::input_left_host_data(run),
                TconT::input_right_host_data(run));
            validation = verification_result.passed ? "PASS" : "FAIL";
        }

        printf(
            "TCCG_RESULT equation=%d precision=%s operations=%ld time_ms=%.4f gflops=%.4f validation=%s\n",
            target_tccg_benchmark,
            desc.scalar_type == TconT::ScalarType::Float32 ? "tf32" : "fp64",
            operation_count,
            result.avg_ms,
            result.gflops,
            validation.c_str());
        return 0;
    } catch (const std::exception& ex) {
        fprintf(stderr, "[TconT] Fatal error: %s\n", ex.what());
        return 2;
    } catch (...) {
        fprintf(stderr, "[TconT] Fatal error: unknown exception\n");
        return 3;
    }
    
    return 0;
}
