#include "cogent.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include "../helper/compile.cuh"
#include "../helper/tcl_helper.h"
#include "../helper/cuda_helper.h"

namespace cogent
{
namespace
{
constexpr const char* kBackendRoot = "backends/cogent";

std::vector<int64_t> collect_extents(
    const std::vector<char>& modes,
    const std::unordered_map<char, int64_t>& extents)
{
    std::vector<int64_t> values;
    values.reserve(modes.size());

    for (char mode : modes) {
        const auto it = extents.find(mode);
        if (it == extents.end()) {
            throw std::invalid_argument(std::string("Missing extent for mode: ") + mode);
        }
        values.push_back(it->second);
    }

    return values;
}

size_t count_elements(
    const std::vector<char>& modes,
    const std::unordered_map<char, int64_t>& extents)
{
    size_t total = 1;
    for (char mode : modes) {
        const auto it = extents.find(mode);
        if (it == extents.end()) {
            throw std::invalid_argument(std::string("Missing extent for mode: ") + mode);
        }
        total *= static_cast<size_t>(it->second);
    }
    return total;
}

int lookup_extent(char mode, const std::unordered_map<char, int64_t>& extents)
{
    const auto it = extents.find(mode);
    if (it == extents.end()) {
        throw std::invalid_argument(std::string("Missing extent for mode: ") + mode);
    }
    return static_cast<int>(it->second);
}

std::string run_planner(const nlohmann::json& input)
{
    const std::string cmd =
        "python3 " + std::string(kBackendRoot) + "/plan/tc_plan.py '" + input.dump() + "'";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (pipe == nullptr) {
        throw std::runtime_error("Failed to start Cogent planner");
    }

    char buffer[4096];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }

    const int status = pclose(pipe);
    if (status != 0) {
        throw std::runtime_error("Cogent planner failed with exit code " + std::to_string(status));
    }

    return result;
}

void ensure_cuda_driver_ready()
{
    CUresult init_result = cuInit(0);
    if (init_result != CUDA_SUCCESS) {
        const char* error_string = nullptr;
        cuGetErrorString(init_result, &error_string);
        throw std::runtime_error(
            std::string("cuInit failed: ") +
            (error_string != nullptr ? error_string : "unknown"));
    }

    cudaError_t runtime_result = cudaFree(nullptr);
    if (runtime_result != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaFree(nullptr) failed while creating CUDA runtime context: ") +
            cudaGetErrorString(runtime_result));
    }
}

void ensure_cubin_exists(const KernelConfig& cfg)
{
    namespace fs = std::filesystem;

    if (fs::exists(cfg.kernel_bin)) {
        return;
    }

    fs::path cubin_path(cfg.kernel_bin);
    fs::path cu_path = fs::path(kBackendRoot) / "code" / cubin_path.filename();
    cu_path.replace_extension(".cu");

    if (!fs::exists(cu_path)) {
        throw std::runtime_error("Neither .cubin nor .cu found: " + cu_path.string());
    }

    compile_to_cubin(cu_path.string(), cubin_path.string());
}

PreparedKernel load_kernel(KernelConfig cfg)
{
    ensure_cuda_driver_ready();
    ensure_cubin_exists(cfg);

    PreparedKernel prepared;
    prepared.config = std::move(cfg);
    prepared.shared_mem_bytes = static_cast<unsigned int>(
        TconT::scalar_type_size(prepared.config.scalar_type) *
        static_cast<size_t>(prepared.config.smem_x + prepared.config.smem_y) *
        static_cast<size_t>(prepared.config.stage));

    CUresult load_result = cuModuleLoad(&prepared.module, prepared.config.kernel_bin.c_str());
    if (load_result != CUDA_SUCCESS) {
        const char* error_string = nullptr;
        cuGetErrorString(load_result, &error_string);
        throw std::runtime_error(
            std::string("cuModuleLoad failed: ") +
            (error_string != nullptr ? error_string : "unknown") +
            " / path: " + prepared.config.kernel_bin);
    }

    CUresult function_result =
        cuModuleGetFunction(&prepared.function, prepared.module, prepared.config.kernel_name.c_str());
    if (function_result != CUDA_SUCCESS) {
        const char* error_string = nullptr;
        cuGetErrorString(function_result, &error_string);
        cuModuleUnload(prepared.module);
        prepared.module = nullptr;
        throw std::runtime_error(
            std::string("cuModuleGetFunction failed: ") +
            prepared.config.kernel_name + " / " +
            (error_string != nullptr ? error_string : "unknown"));
    }

    return prepared;
}

class CogentRunImpl final : public TconT::RunImpl {
public:
    CogentRunImpl(std::shared_ptr<const PreparedKernel> kernel, const TconT::TCEquation& desc)
        : kernel_(std::move(kernel))
    {
        const size_t elements_a = count_elements(desc.modeA, desc.extent);
        const size_t elements_b = count_elements(desc.modeB, desc.extent);
        const size_t elements_c = count_elements(desc.modeC, desc.extent);

        switch (desc.scalar_type) {
            case TconT::ScalarType::Float32:
                prepare_buffers<float>(elements_a, elements_b, elements_c);
                break;
            case TconT::ScalarType::Float64:
                prepare_buffers<double>(elements_a, elements_b, elements_c);
                break;
        }

        size_internal_ = kernel_->config.size_internal;
        smem_y_ = kernel_->config.smem_y;
        smem_x_ = kernel_->config.smem_x;
        external_sizes_ = kernel_->external_sizes;
        internal_sizes_ = kernel_->internal_sizes;
        ceils_ = kernel_->ceils;

        args_.reserve(
            3 + external_sizes_.size() + internal_sizes_.size() + ceils_.size() + 3);
        args_.push_back(&device_c_raw_);
        if (kernel_->config.swap_flag) {
            args_.push_back(&device_b_raw_);
            args_.push_back(&device_a_raw_);
        } else {
            args_.push_back(&device_a_raw_);
            args_.push_back(&device_b_raw_);
        }

        for (int& size : external_sizes_) {
            args_.push_back(&size);
        }
        for (int& size : internal_sizes_) {
            args_.push_back(&size);
        }
        for (int& ceil_value : ceils_) {
            args_.push_back(&ceil_value);
        }

        args_.push_back(&size_internal_);
        args_.push_back(&smem_y_);
        args_.push_back(&smem_x_);
    }

    TconT::Backend backend() const override
    {
        return TconT::Backend::COGENT;
    }

    void launch() override
    {
        const CUresult launch_result = cuLaunchKernel(
            kernel_->function,
            kernel_->grid_dim_x,
            1,
            1,
            kernel_->block_dim_x,
            1,
            1,
            kernel_->shared_mem_bytes,
            nullptr,
            args_.data(),
            nullptr);

        if (launch_result != CUDA_SUCCESS) {
            const char* error_string = nullptr;
            cuGetErrorString(launch_result, &error_string);
            throw std::runtime_error(
                std::string("cuLaunchKernel failed: ") +
                (error_string != nullptr ? error_string : "unknown"));
        }
    }

    const void* input_left_host_data() const override
    {
        return host_a_storage_.data();
    }

    const void* input_right_host_data() const override
    {
        return host_b_storage_.data();
    }

    void copy_output_to_host(void* destination, size_t bytes) const override
    {
        HANDLE_CUDA_ERROR(cudaMemcpy(destination, device_c_.get(), bytes, cudaMemcpyDeviceToHost));
    }

private:
    struct DeviceDeleter {
        void operator()(void* ptr) const
        {
            if (ptr != nullptr) {
                cudaFree(ptr);
            }
        }
    };

    template <typename ValueType>
    void prepare_buffers(size_t elements_a, size_t elements_b, size_t elements_c)
    {
        host_a_storage_.resize(sizeof(ValueType) * elements_a);
        host_b_storage_.resize(sizeof(ValueType) * elements_b);
        host_c_storage_.assign(sizeof(ValueType) * elements_c, 0);

        ValueType* host_a = reinterpret_cast<ValueType*>(host_a_storage_.data());
        ValueType* host_b = reinterpret_cast<ValueType*>(host_b_storage_.data());

        pre_Initializing_Input_Tensors(
            host_a,
            static_cast<int>(elements_a),
            host_b,
            static_cast<int>(elements_b));

        void* dA = nullptr;
        void* dB = nullptr;
        void* dC = nullptr;
        HANDLE_CUDA_ERROR(cudaMalloc(&dA, sizeof(ValueType) * elements_a));
        HANDLE_CUDA_ERROR(cudaMalloc(&dB, sizeof(ValueType) * elements_b));
        HANDLE_CUDA_ERROR(cudaMalloc(&dC, sizeof(ValueType) * elements_c));

        device_a_.reset(dA);
        device_b_.reset(dB);
        device_c_.reset(dC);
        device_a_raw_ = device_a_.get();
        device_b_raw_ = device_b_.get();
        device_c_raw_ = device_c_.get();

        HANDLE_CUDA_ERROR(cudaMemcpy(device_a_.get(), host_a, sizeof(ValueType) * elements_a, cudaMemcpyHostToDevice));
        HANDLE_CUDA_ERROR(cudaMemcpy(device_b_.get(), host_b, sizeof(ValueType) * elements_b, cudaMemcpyHostToDevice));
        HANDLE_CUDA_ERROR(cudaMemset(device_c_.get(), 0, sizeof(ValueType) * elements_c));
    }

    std::shared_ptr<const PreparedKernel> kernel_;
    std::unique_ptr<void, DeviceDeleter> device_a_;
    std::unique_ptr<void, DeviceDeleter> device_b_;
    std::unique_ptr<void, DeviceDeleter> device_c_;
    void* device_a_raw_ = nullptr;
    void* device_b_raw_ = nullptr;
    void* device_c_raw_ = nullptr;
    std::vector<unsigned char> host_a_storage_;
    std::vector<unsigned char> host_b_storage_;
    std::vector<unsigned char> host_c_storage_;
    std::vector<int> external_sizes_;
    std::vector<int> internal_sizes_;
    std::vector<int> ceils_;
    int size_internal_ = 0;
    int smem_y_ = 0;
    int smem_x_ = 0;
    std::vector<void*> args_;
};

class CogentPlanImpl final : public TconT::PlanImpl {
public:
    explicit CogentPlanImpl(PreparedKernel prepared_kernel, TconT::TCEquation equation)
        : kernel_(std::make_shared<PreparedKernel>(std::move(prepared_kernel))),
          equation_(std::move(equation)) {}

    TconT::Backend backend() const override
    {
        return TconT::Backend::COGENT;
    }

    std::shared_ptr<TconT::RunImpl> prepare() const override
    {
        return std::make_shared<CogentRunImpl>(kernel_, equation_);
    }

private:
    std::shared_ptr<PreparedKernel> kernel_;
    TconT::TCEquation equation_;
};
}  // namespace

PreparedKernel::~PreparedKernel()
{
    if (module != nullptr) {
        cuModuleUnload(module);
    }
}

PreparedKernel::PreparedKernel(PreparedKernel&& other) noexcept
    : config(std::move(other.config)),
      module(other.module),
      function(other.function),
      shared_mem_bytes(other.shared_mem_bytes),
      grid_dim_x(other.grid_dim_x),
      block_dim_x(other.block_dim_x),
      external_sizes(std::move(other.external_sizes)),
      internal_sizes(std::move(other.internal_sizes)),
      ceils(std::move(other.ceils))
{
    other.module = nullptr;
    other.function = nullptr;
    other.shared_mem_bytes = 0;
    other.grid_dim_x = 0;
    other.block_dim_x = 0;
}

PreparedKernel& PreparedKernel::operator=(PreparedKernel&& other) noexcept
{
    if (this != &other) {
        if (module != nullptr) {
            cuModuleUnload(module);
        }

        config = std::move(other.config);
        module = other.module;
        function = other.function;
        shared_mem_bytes = other.shared_mem_bytes;
        grid_dim_x = other.grid_dim_x;
        block_dim_x = other.block_dim_x;
        external_sizes = std::move(other.external_sizes);
        internal_sizes = std::move(other.internal_sizes);
        ceils = std::move(other.ceils);

        other.module = nullptr;
        other.function = nullptr;
        other.shared_mem_bytes = 0;
        other.grid_dim_x = 0;
        other.block_dim_x = 0;
    }

    return *this;
}

std::shared_ptr<TconT::PlanImpl> plan_cogent(const TconT::TCEquation& desc)
{
    if (desc.scalar_type != TconT::ScalarType::Float64) {
        throw std::runtime_error(
            std::string("Cogent planner currently supports only double kernels. Requested scalar type: ") +
            TconT::scalar_type_name(desc.scalar_type));
    }

    nlohmann::json input;
    input["modeC"] = desc.modeC;
    input["extentC"] = collect_extents(desc.modeC, desc.extent);
    input["modeA"] = desc.modeA;
    input["extentA"] = collect_extents(desc.modeA, desc.extent);
    input["modeB"] = desc.modeB;
    input["extentB"] = collect_extents(desc.modeB, desc.extent);
    input["op"] = desc.op;
    input["type"] = TconT::scalar_type_name(desc.scalar_type);

    const nlohmann::json output = nlohmann::json::parse(run_planner(input));

    KernelConfig cfg;
    const std::string kernel_bin = output.at("kernel_bin").get<std::string>();

    cfg.scalar_type = desc.scalar_type;
    cfg.kernel_bin = (std::filesystem::path(kBackendRoot) / "bin" / (kernel_bin + ".cubin")).string();
    cfg.kernel_name = output.at("kernel_name").get<std::string>();
    cfg.external_index = output.at("external_index").get<std::vector<std::string>>();
    cfg.internal_index = output.at("internal_index").get<std::vector<std::string>>();
    cfg.block_size = output.at("block_size").get<int>();
    cfg.stage = output.at("stage").at(0).get<int>();
    cfg.smem_x = output.at("smem_x").get<int>();
    cfg.smem_y = output.at("smem_y").get<int>();
    cfg.warp_shape = output.at("warp_shape").get<std::vector<int>>();
    cfg.size_internal = output.at("internal").get<int>();
    cfg.swap_flag = output.at("swap_flag").get<bool>();

    for (const auto& tile : output.at("tile_sizes")) {
        cfg.tile_sizes.emplace(tile.at(0).get<std::string>(), tile.at(1).get<int>());
    }

    PreparedKernel prepared = load_kernel(std::move(cfg));
    prepared.block_dim_x = static_cast<unsigned int>(prepared.config.block_size);

    int num_thread_blocks = 1;
    prepared.external_sizes.reserve(prepared.config.external_index.size());
    prepared.ceils.reserve(prepared.config.external_index.size());
    for (const auto& idx : prepared.config.external_index) {
        const int extent = lookup_extent(idx.front(), desc.extent);
        const int ceil_value = (extent + prepared.config.tile_sizes.at(idx) - 1) /
            prepared.config.tile_sizes.at(idx);
        prepared.external_sizes.push_back(extent);
        prepared.ceils.push_back(ceil_value);
        num_thread_blocks *= ceil_value;
    }

    prepared.internal_sizes.reserve(prepared.config.internal_index.size());
    for (const auto& idx : prepared.config.internal_index) {
        prepared.internal_sizes.push_back(lookup_extent(idx.front(), desc.extent));
    }

    prepared.grid_dim_x = static_cast<unsigned int>(num_thread_blocks);

    return std::make_shared<CogentPlanImpl>(std::move(prepared), desc);
}
}  // namespace cogent
