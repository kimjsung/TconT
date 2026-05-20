#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <numeric>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "tcont.hpp"
#include "tccg_cases.hpp"
#include "tccg_utils.hpp"
#include "tccg_verify.hpp"

namespace
{
TconT::BackendSelection parse_backend_selection(const char* value)
{
    if (std::strcmp(value, "model") == 0) {
        return TconT::BackendSelection::MODEL;
    }
    if (std::strcmp(value, "cogent") == 0) {
        return TconT::BackendSelection::COGENT;
    }
    if (std::strcmp(value, "ttgt") == 0 || std::strcmp(value, "ttgt_cutt") == 0) {
        return TconT::BackendSelection::TTGT_CUTT;
    }
    throw std::invalid_argument(std::string("Unknown backend: ") + value);
}
}

int main(int argc, char *argv[])
{
    try
    {
        int target_tccg_benchmark = -1;
        bool verify = false;
        TconT::BackendSelection backend_selection = TconT::BackendSelection::MODEL;
        const std::vector<TconT::TCEquation>* benchmark_cases = &list_tccg_bench;

        for (int i = 1; i < argc; ++i)
        {
            const char* arg = argv[i];

            if (std::strcmp(arg, "-b") == 0 || std::strcmp(arg, "--benchmark") == 0)
            {
                if (i + 1 < argc) {
                    target_tccg_benchmark = std::atoi(argv[++i]);
                }
            }
            else if (std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--help") == 0)
            {
                std::printf("Usage: %s [-b <benchmark_id>] [--tf32|--fp32|--fp64] [--backend <name>] [--verify]\n", argv[0]);
                std::printf("  -b, --benchmark <id>       Specify TCCG benchmark ID (0-48)\n");
                std::printf("      --tf32                 Run the TF32 benchmark cases\n");
                std::printf("      --fp32                 Alias for --tf32 (backward compatibility)\n");
                std::printf("      --fp64                 Run the FP64 benchmark cases (default)\n");
                std::printf("      --backend <name>       Select backend: model, cogent, or ttgt\n");
                std::printf("      --verify               Run correctness verification after timing\n");
                return 0;
            }
            else if (std::strcmp(arg, "--backend") == 0)
            {
                if (i + 1 < argc) {
                    backend_selection = parse_backend_selection(argv[++i]);
                }
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

        if (target_tccg_benchmark < 0 || target_tccg_benchmark >= static_cast<int>(benchmark_cases->size())) {
            std::printf("Invalid benchmark ID: %d. Valid range is 0 to %lu.\n", target_tccg_benchmark, benchmark_cases->size() - 1);
            return -1;
        }

        TconT::TCEquation desc = (*benchmark_cases)[static_cast<size_t>(target_tccg_benchmark)];
        desc.backend_selection = backend_selection;
        const int64_t operation_count = 2 * static_cast<int64_t>(
            std::sqrt(
                static_cast<double>(compute_tensor_size(desc.modeA, desc.extent)) *
                static_cast<double>(compute_tensor_size(desc.modeB, desc.extent)) *
                static_cast<double>(compute_tensor_size(desc.modeC, desc.extent))));

        std::printf("-----------------------------------------------------------------------\n");
        std::printf("[TconT] Target TCCG's Benchmark: %d\n", target_tccg_benchmark);
        std::printf("[TconT] column-major format.\n");

        print_equation(desc);

        const int64_t size_A = compute_tensor_size(desc.modeA, desc.extent);
        const int64_t size_B = compute_tensor_size(desc.modeB, desc.extent);
        const int64_t size_C = compute_tensor_size(desc.modeC, desc.extent);

        const size_t scalar_size = TconT::scalar_type_size(desc.scalar_type);
        std::printf("[TconT] scalar type: %s\n", TconT::scalar_type_name(desc.scalar_type));
        std::printf("[TconT] selection  : %s\n", TconT::backend_selection_name(desc.backend_selection));
        std::printf("[TconT]  Size A: %12ld (words), %16ld (bytes)\n", size_A, size_A * scalar_size);
        std::printf("[TconT]  Size B: %12ld (words), %16ld (bytes)\n", size_B, size_B * scalar_size);
        std::printf("[TconT]  Size C: %12ld (words), %16ld (bytes)\n", size_C, size_C * scalar_size);
        std::printf("[TconT] # FLOPS: %20ld\n", operation_count);
        std::printf("-----------------------------------------------------------------------\n");

        TconT::ExecutionPlan plan = TconT::plan(desc);
        std::printf("[TconT] backend    : %s\n", TconT::backend_name(plan.backend));
        const TconT::ExecutionRun run = TconT::prepare(plan);
        const TconT::RunOptions options{100, 200};

        for (int i = 0; i < options.warmup; ++i) {
            TconT::launch(run);
        }
        TconT::check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)");

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        TconT::check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
        TconT::check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

        float total_ms = 0.0f;
        try {
            TconT::check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
            for (int i = 0; i < options.repeats; ++i) {
                TconT::launch(run);
            }
            TconT::check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
            TconT::check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
            TconT::check_cuda(cudaEventElapsedTime(&total_ms, start, stop), "cudaEventElapsedTime");
            TconT::check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
            TconT::check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
        } catch (...) {
            if (start != nullptr) {
                cudaEventDestroy(start);
            }
            if (stop != nullptr) {
                cudaEventDestroy(stop);
            }
            throw;
        }

        const float avg_ms = total_ms / static_cast<float>(options.repeats);
        const TconT::ContractionResult result{
            avg_ms,
            TconT::compute_gflops(plan.equation, avg_ms),
            true,
        };

        std::printf("[TconT] Avg Time: %10.4f ms\n", result.avg_ms);
        std::printf("[TconT] GFLOPS  : %10.4f GFLOPS\n", result.gflops);

        std::string validation = "SKIP";
        if (verify) {
            TconT::launch(run);
            TconT::check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(verify)");

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

        std::printf(
            "TCCG_RESULT equation=%d precision=%s backend=%s operations=%ld time_ms=%.4f gflops=%.4f validation=%s\n",
            target_tccg_benchmark,
            desc.scalar_type == TconT::ScalarType::Float32 ? "tf32" : "fp64",
            TconT::backend_name(plan.backend),
            operation_count,
            result.avg_ms,
            result.gflops,
            validation.c_str());
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[TconT] Fatal error: %s\n", ex.what());
        return 2;
    } catch (...) {
        std::fprintf(stderr, "[TconT] Fatal error: unknown exception\n");
        return 3;
    }
}
