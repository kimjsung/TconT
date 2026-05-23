#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

#include <cuda_runtime.h>

#include "tcont.hpp"
#include "cogent.hpp"
#include "ttgt_cutt_backend.hpp"

namespace TconT
{
    namespace
    {
        Backend select_backend_by_model(const TCEquation& desc)
        {
            (void)desc;
            std::cout << "[TconT] Model selection is not implemented yet; falling back to Cogent" << std::endl;
            return Backend::COGENT;
        }
    }

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

    ExecutionPlan plan(const TCEquation& desc) {
        ExecutionPlan execution_plan;
        Backend selected_backend = Backend::COGENT;

        switch (desc.backend_selection) {
            case BackendSelection::MODEL:
                selected_backend = select_backend_by_model(desc);
                break;
            case BackendSelection::COGENT:
                selected_backend = Backend::COGENT;
                break;
            case BackendSelection::TTGT_CUTT:
                selected_backend = Backend::TTGT_CUTT;
                break;
        }

        switch (selected_backend) {
            case Backend::COGENT:
                std::cout << "[TconT] Cogent selected" << std::endl;
                execution_plan.impl = cogent::plan_cogent(desc);
                break;
            case Backend::TTGT_CUTT:
                std::cout << "[TconT] TTGT selected" << std::endl;
                execution_plan.impl = ttgt_cutt_backend::plan_ttgt_cutt(desc);
                break;
        }

        execution_plan.backend = selected_backend;
        execution_plan.equation = desc;
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

    void warmup(const ExecutionRun& run, int iterations) {
        if (!run.impl) {
            throw std::invalid_argument("ExecutionRun is empty");
        }
        if (iterations <= 0) {
            return;
        }
        if (run.impl->warmup(iterations)) {
            return;
        }
        for (int i = 0; i < iterations; ++i) {
            run.impl->launch();
        }
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)");
    }

    void zero_output(const ExecutionRun& run) {
        if (!run.impl) {
            throw std::invalid_argument("ExecutionRun is empty");
        }

        run.impl->zero_output();
    }

    ExecutionStageTimes stage_times(const ExecutionRun& run) {
        if (!run.impl) {
            throw std::invalid_argument("ExecutionRun is empty");
        }

        return run.impl->stage_times();
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
}
