#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace TconT
{
    enum class ScalarType {
        Float32,
        Float64
    };

    constexpr size_t scalar_type_size(ScalarType type)
    {
        switch (type) {
            case ScalarType::Float32:
                return 4;
            case ScalarType::Float64:
                return 8;
        }
        return 0;
    }

    constexpr const char* scalar_type_name(ScalarType type)
    {
        switch (type) {
            case ScalarType::Float32:
                return "FLOAT";
            case ScalarType::Float64:
                return "DOUBLE";
        }
        return "unknown";
    }

    enum class Backend { 
        COGENT, 
        TTGT_CUTT
    };
    
    struct TCEquation {
        std::vector<char> modeC;
        std::vector<char> modeA;
        std::vector<char> modeB;
        char op = '?';
        ScalarType scalar_type = ScalarType::Float64;

        std::unordered_map<char, int64_t> extent;
    };

    struct ContractionResult {
        float avg_ms;
        double gflops;
        bool coorect;
    };

    struct RunOptions {
        int warmup = 3;
        int repeats = 10;
    };

    struct RunImpl;

    struct PlanImpl {
        virtual ~PlanImpl() = default;
        virtual Backend backend() const = 0;
        virtual std::shared_ptr<RunImpl> prepare() const = 0;
    };

    struct RunImpl {
        virtual ~RunImpl() = default;
        virtual Backend backend() const = 0;
        virtual void launch() = 0;
        virtual const void* input_left_host_data() const = 0;
        virtual const void* input_right_host_data() const = 0;
        virtual void copy_output_to_host(void* destination, size_t bytes) const = 0;
    };

    struct ExecutionPlan {
        Backend backend = Backend::COGENT;
        TCEquation equation;
        std::shared_ptr<PlanImpl> impl;
    };

    struct ExecutionRun {
        Backend backend = Backend::COGENT;
        TCEquation equation;
        std::shared_ptr<RunImpl> impl;
    };

    ExecutionPlan plan(const TCEquation& desc);
    ExecutionRun prepare(const ExecutionPlan& plan);
    void launch(const ExecutionRun& run);
    const void* input_left_host_data(const ExecutionRun& run);
    const void* input_right_host_data(const ExecutionRun& run);
    void copy_output_to_host(const ExecutionRun& run, void* destination, size_t bytes);
    ContractionResult benchmark(const ExecutionRun& run, const TCEquation& equation, const RunOptions& options = {});
    ContractionResult benchmark(const ExecutionPlan& plan, const RunOptions& options = {});
    ContractionResult contract(const ExecutionPlan& plan);
    ContractionResult contract(const TCEquation& desc);
}
