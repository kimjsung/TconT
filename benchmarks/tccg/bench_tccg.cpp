#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "tcont.hpp"
#include "tccg_cases.hpp"
#include "tccg_utils.hpp"
#include "tccg_verify.hpp"

int main(int argc, char *argv[])
{
    int target_tccg_benchmark;
    bool verify = false;
    
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
            printf("Usage: %s [-b <benchmark_id>] [--verify]\n", argv[0]);
            printf("  -b, --benchmark <id>       Specify TCCG benchmark ID (1-48)\n");
            printf("      --verify               Run correctness verification after timing\n");
            return 0;
        }
        else if (std::strcmp(arg, "--verify") == 0)
        {
            verify = true;
        }
    }

    if(target_tccg_benchmark < 0 || target_tccg_benchmark >= list_tccg_bench.size()) {
        printf("Invalid benchmark ID: %d. Valid range is 0 to %lu.\n", target_tccg_benchmark, list_tccg_bench.size() - 1);
        return -1;
    }

    printf("-----------------------------------------------------------------------\n");
    printf("[TconT] Target TCCG's Benchmark: %d\n", target_tccg_benchmark);
    printf("[TconT] column-major format.\n");
    
    print_equation(list_tccg_bench[target_tccg_benchmark]);

    int64_t size_A = compute_tensor_size(list_tccg_bench[target_tccg_benchmark].modeA, list_tccg_bench[target_tccg_benchmark].extent);
    int64_t size_B = compute_tensor_size(list_tccg_bench[target_tccg_benchmark].modeB, list_tccg_bench[target_tccg_benchmark].extent);
    int64_t size_C = compute_tensor_size(list_tccg_bench[target_tccg_benchmark].modeC, list_tccg_bench[target_tccg_benchmark].extent);

    const size_t scalar_size = TconT::scalar_type_size(list_tccg_bench[target_tccg_benchmark].scalar_type);
    printf("[TconT] scalar type: %s\n", TconT::scalar_type_name(list_tccg_bench[target_tccg_benchmark].scalar_type));
    printf("[TconT]  Size A: %12ld (words), %16ld (bytes)\n", size_A, size_A * scalar_size);
    printf("[TconT]  Size B: %12ld (words), %16ld (bytes)\n", size_B, size_B * scalar_size);
    printf("[TconT]  Size C: %12ld (words), %16ld (bytes)\n", size_C, size_C * scalar_size);
    printf("[TconT] # FLOPS: %20ld\n", 2 * (int64_t)std::sqrt(size_A * size_B * size_C));
    printf("-----------------------------------------------------------------------\n");

    printf("[TconT] Selct TTGT or Direct\n");
    TconT::TCEquation desc = list_tccg_bench[target_tccg_benchmark];
    
    TconT::ExecutionPlan plan = TconT::plan(desc);
    TconT::ExecutionRun run = TconT::prepare(plan);
    TconT::RunOptions options;
    options.warmup = 3;
    options.repeats = 10;

    TconT::ContractionResult result = TconT::benchmark(run, desc, options);
    
    printf("[TconT] Avg Time: %10.4f ms\n", result.avg_ms);
    printf("[TconT] GFLOPS  : %10.4f GFLOPS\n", result.gflops);

    if (verify) {
        const size_t output_elements = static_cast<size_t>(compute_tensor_size(desc.modeC, desc.extent));
        const size_t output_bytes = output_elements * TconT::scalar_type_size(desc.scalar_type);
        std::vector<unsigned char> output_storage(output_bytes);

        TconT::copy_output_to_host(run, output_storage.data(), output_bytes);
        verify_tccg_case(
            static_cast<size_t>(target_tccg_benchmark),
            desc,
            output_storage.data(),
            TconT::input_left_host_data(run),
            TconT::input_right_host_data(run));
    }
}
