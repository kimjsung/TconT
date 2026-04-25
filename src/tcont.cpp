#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

#include <cuda_runtime.h>

#include "tcont.hpp"
#include "cogent.hpp"

namespace TconT
{
    namespace
    {
        void check_cuda(cudaError_t status, const char* expr)
        {
            if (status != cudaSuccess) {
                throw std::runtime_error(
                    std::string("CUDA error in ") + expr + ": " + cudaGetErrorString(status));
            }
        }

        double compute_gflops(const TCEquation& equation, float avg_ms)
        {
            if (avg_ms <= 0.0f) {
                return 0.0;
            }

            const auto count_elements = [&](const std::vector<char>& modes) {
                double total = 1.0;
                for (char mode : modes) {
                    const auto it = equation.extent.find(mode);
                    if (it == equation.extent.end()) {
                        throw std::invalid_argument(std::string("Missing extent for mode: ") + mode);
                    }
                    total *= static_cast<double>(it->second);
                }
                return total;
            };

            const double elements_a = count_elements(equation.modeA);
            const double elements_b = count_elements(equation.modeB);
            const double elements_c = count_elements(equation.modeC);
            const double flops = 2.0 * std::sqrt(elements_a * elements_b * elements_c);
            return flops / (static_cast<double>(avg_ms) * 1.0e6);
        }
    }

    ExecutionPlan plan(const TCEquation& desc) {
        ExecutionPlan execution_plan;

        // approach selection by model
        // model()

        std::cout << "[TconT] Direct Selected" << std::endl;
        execution_plan.backend = Backend::COGENT;
        execution_plan.equation = desc;
        execution_plan.impl = cogent::plan_cogent(desc);
        
        return execution_plan;
    }

    ExecutionRun prepare(const ExecutionPlan& plan) {
        if (!plan.impl) {
            throw std::invalid_argument("ExecutionPlan is empty");
        }

        ExecutionRun run;
        run.backend = plan.backend;
        run.equation = plan.equation;
        run.impl = plan.impl->prepare();
        return run;
    }

    void launch(const ExecutionRun& run) {
        if (!run.impl) {
            throw std::invalid_argument("ExecutionRun is empty");
        }

        run.impl->launch();
    }

    const void* input_left_host_data(const ExecutionRun& run) {
        if (!run.impl) {
            throw std::invalid_argument("ExecutionRun is empty");
        }

        return run.impl->input_left_host_data();
    }

    const void* input_right_host_data(const ExecutionRun& run) {
        if (!run.impl) {
            throw std::invalid_argument("ExecutionRun is empty");
        }

        return run.impl->input_right_host_data();
    }

    void copy_output_to_host(const ExecutionRun& run, void* destination, size_t bytes) {
        if (!run.impl) {
            throw std::invalid_argument("ExecutionRun is empty");
        }
        if (destination == nullptr) {
            throw std::invalid_argument("Destination buffer is null");
        }

        run.impl->copy_output_to_host(destination, bytes);
    }

    ContractionResult benchmark(const ExecutionRun& run, const TCEquation& equation, const RunOptions& options) {
        if (!run.impl) {
            throw std::invalid_argument("ExecutionRun is empty");
        }
        if (options.warmup < 0 || options.repeats <= 0) {
            throw std::invalid_argument("RunOptions must satisfy warmup >= 0 and repeats > 0");
        }

        for (int i = 0; i < options.warmup; ++i) {
            launch(run);
        }
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

        try {
            check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
            for (int i = 0; i < options.repeats; ++i) {
                launch(run);
            }
            check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
            check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

            float total_ms = 0.0f;
            check_cuda(cudaEventElapsedTime(&total_ms, start, stop), "cudaEventElapsedTime");
            check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
            check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

            const float avg_ms = total_ms / static_cast<float>(options.repeats);

            
            return ContractionResult{avg_ms, compute_gflops(equation, avg_ms), true};
        } catch (...) {
            if (start != nullptr) {
                cudaEventDestroy(start);
            }
            if (stop != nullptr) {
                cudaEventDestroy(stop);
            }
            throw;
        }
    }

    ContractionResult benchmark(const ExecutionPlan& plan, const RunOptions& options) {
        const ExecutionRun run = prepare(plan);
        return benchmark(run, plan.equation, options);
    }

    ContractionResult contract(const ExecutionPlan& plan) {
        return benchmark(plan, RunOptions{0, 1});
    }

    ContractionResult contract(const TCEquation& desc) {
        return contract(plan(desc));
    }
}
