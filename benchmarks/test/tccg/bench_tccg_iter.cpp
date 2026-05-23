#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
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

struct CliOptions {
    int benchmark_id = -1;
    bool verify = false;
    TconT::BackendSelection backend_selection = TconT::BackendSelection::MODEL;
    const std::vector<TconT::TCEquation>* benchmark_cases = &list_tccg_bench;
    int warmup = 10;
    int repeats = 200;
    std::string output_path;
};

double compute_stddev_ms(const std::vector<float>& samples, double avg_ms)
{
    if (samples.empty()) {
        return 0.0;
    }

    double accum = 0.0;
    for (float value : samples) {
        const double diff = static_cast<double>(value) - avg_ms;
        accum += diff * diff;
    }

    return std::sqrt(accum / static_cast<double>(samples.size()));
}

void print_usage(const char* argv0)
{
    std::printf(
        "Usage: %s -b <benchmark_id> [--backend <name>] [--warmup <n>] [--repeats <n>] "
        "[--tf32|--fp32|--fp64] [--verify] [--output <path>]\n",
        argv0);
}

CliOptions parse_args(int argc, char* argv[])
{
    CliOptions options;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];

        if (std::strcmp(arg, "-b") == 0 || std::strcmp(arg, "--benchmark") == 0) {
            if (i + 1 < argc) {
                options.benchmark_id = std::atoi(argv[++i]);
            }
        } else if (std::strcmp(arg, "--backend") == 0) {
            if (i + 1 < argc) {
                options.backend_selection = parse_backend_selection(argv[++i]);
            }
        } else if (std::strcmp(arg, "--warmup") == 0) {
            if (i + 1 < argc) {
                options.warmup = std::atoi(argv[++i]);
            }
        } else if (std::strcmp(arg, "--repeats") == 0) {
            if (i + 1 < argc) {
                options.repeats = std::atoi(argv[++i]);
            }
        } else if (std::strcmp(arg, "--verify") == 0) {
            options.verify = true;
        } else if (std::strcmp(arg, "--tf32") == 0 || std::strcmp(arg, "--fp32") == 0) {
            options.benchmark_cases = &list_tccg_bench_fp32;
        } else if (std::strcmp(arg, "--fp64") == 0) {
            options.benchmark_cases = &list_tccg_bench;
        } else if (std::strcmp(arg, "--output") == 0) {
            if (i + 1 < argc) {
                options.output_path = argv[++i];
            }
        } else if (std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument(std::string("Unknown argument: ") + arg);
        }
    }

    if (options.benchmark_id < 0 || options.benchmark_id >= static_cast<int>(options.benchmark_cases->size())) {
        throw std::invalid_argument("Benchmark ID must be within the valid TCCG range");
    }
    if (options.warmup < 0) {
        throw std::invalid_argument("Warmup must be >= 0");
    }
    if (options.repeats <= 0) {
        throw std::invalid_argument("Repeats must be > 0");
    }

    return options;
}
}

int main(int argc, char* argv[])
{
    try {
        const CliOptions cli = parse_args(argc, argv);

        TconT::TCEquation desc = (*cli.benchmark_cases)[static_cast<size_t>(cli.benchmark_id)];
        desc.backend_selection = cli.backend_selection;

        const int64_t operation_count = 2 * static_cast<int64_t>(
            std::sqrt(
                static_cast<double>(compute_tensor_size(desc.modeA, desc.extent)) *
                static_cast<double>(compute_tensor_size(desc.modeB, desc.extent)) *
                static_cast<double>(compute_tensor_size(desc.modeC, desc.extent))));

        std::printf("-----------------------------------------------------------------------\n");
        std::printf("[TconT] Iteration benchmark for TCCG equation %d\n", cli.benchmark_id);
        std::printf("[TconT] column-major format.\n");
        print_equation(desc);
        std::printf("[TconT] scalar type: %s\n", TconT::scalar_type_name(desc.scalar_type));
        std::printf("[TconT] selection  : %s\n", TconT::backend_selection_name(desc.backend_selection));
        std::printf("[TconT] warmup     : %d\n", cli.warmup);
        std::printf("[TconT] repeats    : %d\n", cli.repeats);
        std::printf("-----------------------------------------------------------------------\n");

        const TconT::ExecutionPlan plan = TconT::plan(desc);
        const TconT::ExecutionRun run = TconT::prepare(plan);

        TconT::warmup(run, cli.warmup);
        TconT::zero_output(run);

        std::ofstream csv_file;
        if (!cli.output_path.empty()) {
            csv_file.open(cli.output_path);
            if (!csv_file) {
                throw std::runtime_error("Failed to open output file: " + cli.output_path);
            }
            csv_file << "iteration,time_ms,gflops,stage_total_ms,transpose_a_ms,transpose_b_ms,gemm_ms,transpose_c_ms\n";
        }

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        TconT::check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
        TconT::check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

        std::vector<float> iteration_ms;
        iteration_ms.reserve(static_cast<size_t>(cli.repeats));

        try {
            for (int iter = 0; iter < cli.repeats; ++iter) {
                TconT::check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
                TconT::launch(run);
                TconT::check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
                TconT::check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

                float elapsed_ms = 0.0f;
                TconT::check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
                iteration_ms.push_back(elapsed_ms);
                const TconT::ExecutionStageTimes stage_times = TconT::stage_times(run);

                const double iteration_gflops = TconT::compute_gflops(plan.equation, elapsed_ms);
                std::printf(
                    "ITERATION %04d time_ms=%.6f gflops=%.6f stage_total_ms=%.6f transpose_a_ms=%.6f transpose_b_ms=%.6f gemm_ms=%.6f transpose_c_ms=%.6f\n",
                    iter,
                    elapsed_ms,
                    iteration_gflops,
                    stage_times.total_ms,
                    stage_times.transpose_a_ms,
                    stage_times.transpose_b_ms,
                    stage_times.gemm_ms,
                    stage_times.transpose_c_ms);
                if (csv_file) {
                    csv_file << iter
                             << ',' << elapsed_ms
                             << ',' << iteration_gflops
                             << ',' << stage_times.total_ms
                             << ',' << stage_times.transpose_a_ms
                             << ',' << stage_times.transpose_b_ms
                             << ',' << stage_times.gemm_ms
                             << ',' << stage_times.transpose_c_ms
                             << '\n';
                }
            }
        } catch (...) {
            if (start != nullptr) {
                cudaEventDestroy(start);
            }
            if (stop != nullptr) {
                cudaEventDestroy(stop);
            }
            throw;
        }

        TconT::check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
        TconT::check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

        const double total_ms = std::accumulate(iteration_ms.begin(), iteration_ms.end(), 0.0);
        const double avg_ms = total_ms / static_cast<double>(iteration_ms.size());

        float min_ms = std::numeric_limits<float>::max();
        float max_ms = 0.0f;
        for (float sample : iteration_ms) {
            if (sample < min_ms) {
                min_ms = sample;
            }
            if (sample > max_ms) {
                max_ms = sample;
            }
        }

        const double stddev_ms = compute_stddev_ms(iteration_ms, avg_ms);
        const double avg_gflops = TconT::compute_gflops(plan.equation, static_cast<float>(avg_ms));

        std::string validation = "SKIP";
        if (cli.verify) {
            const size_t output_elements = static_cast<size_t>(compute_tensor_size(desc.modeC, desc.extent));
            const size_t output_bytes = output_elements * TconT::scalar_type_size(desc.scalar_type);
            std::vector<unsigned char> output_storage(output_bytes);

            TconT::copy_output_to_host(run, output_storage.data(), output_bytes);
            const VerificationResult verification_result = verify_tccg_case(
                static_cast<size_t>(cli.benchmark_id),
                desc,
                output_storage.data(),
                TconT::input_left_host_data(run),
                TconT::input_right_host_data(run));
            validation = verification_result.passed ? "PASS" : "FAIL";
        }

        std::printf("-----------------------------------------------------------------------\n");
        std::printf("[TconT] backend    : %s\n", TconT::backend_name(plan.backend));
        std::printf("[TconT] Avg Time   : %10.6f ms\n", avg_ms);
        std::printf("[TconT] Min Time   : %10.6f ms\n", min_ms);
        std::printf("[TconT] Max Time   : %10.6f ms\n", max_ms);
        std::printf("[TconT] Stddev     : %10.6f ms\n", stddev_ms);
        std::printf("[TconT] Avg GFLOPS : %10.6f GFLOPS\n", avg_gflops);
        std::printf(
            "TCCG_ITER_RESULT equation=%d precision=%s backend=%s operations=%ld repeats=%d avg_ms=%.6f min_ms=%.6f "
            "max_ms=%.6f stddev_ms=%.6f gflops=%.6f validation=%s\n",
            cli.benchmark_id,
            desc.scalar_type == TconT::ScalarType::Float32 ? "tf32" : "fp64",
            TconT::backend_name(plan.backend),
            operation_count,
            cli.repeats,
            avg_ms,
            min_ms,
            max_ms,
            stddev_ms,
            avg_gflops,
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
