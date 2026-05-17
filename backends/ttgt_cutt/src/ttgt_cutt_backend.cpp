#include "ttgt_cutt_backend.hpp"

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <limits>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "../../cogent/helper/cuda_helper.h"
#include "ttgt_cutt.hpp"

namespace ttgt_cutt_backend
{
namespace
{
void check_cuda_status(cudaError_t status, const char* expr)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error in ") + expr + ": " + cudaGetErrorString(status));
    }
}

bool is_identity_permutation(const std::vector<int>& permutation)
{
    for (size_t i = 0; i < permutation.size(); ++i) {
        if (permutation[i] != static_cast<int>(i)) {
            return false;
        }
    }
    return true;
}

void validate_permutation(const std::vector<int>& permutation, size_t expected_rank, const char* label)
{
    if (permutation.size() != expected_rank) {
        throw std::runtime_error(
            std::string("Invalid TTGT permutation rank for ") + label +
            ": expected " + std::to_string(expected_rank) +
            ", got " + std::to_string(permutation.size()));
    }

    std::vector<int> seen(expected_rank, 0);
    for (int value : permutation) {
        if (value < 0 || value >= static_cast<int>(expected_rank)) {
            throw std::runtime_error(
                std::string("Invalid TTGT permutation value for ") + label +
                ": " + std::to_string(value));
        }
        if (++seen[static_cast<size_t>(value)] > 1) {
            throw std::runtime_error(
                std::string("Duplicated TTGT permutation value for ") + label +
                ": " + std::to_string(value));
        }
    }
}

void fill_dims_from_modes(
    std::vector<int>& dims,
    const std::vector<char>& modes,
    const std::unordered_map<char, int64_t>& extents)
{
    dims.clear();
    dims.reserve(modes.size());
    for (char mode : modes) {
        const auto it = extents.find(mode);
        if (it == extents.end()) {
            throw std::invalid_argument(std::string("Missing extent for mode: ") + mode);
        }
        dims.push_back(static_cast<int>(it->second));
    }
}

std::vector<int> build_transposed_input_dims(
    const std::vector<char>& output_modes,
    const std::unordered_map<char, int64_t>& extents,
    const std::vector<int>& permutation)
{
    validate_permutation(permutation, output_modes.size(), "output");

    std::vector<int> dims(output_modes.size(), 0);
    for (size_t output_idx = 0; output_idx < output_modes.size(); ++output_idx) {
        const char mode = output_modes[output_idx];
        const auto it = extents.find(mode);
        if (it == extents.end()) {
            throw std::invalid_argument(std::string("Missing extent for mode: ") + mode);
        }
        dims[static_cast<size_t>(permutation[output_idx])] = static_cast<int>(it->second);
    }
    return dims;
}

bool is_external_mode(const TconT::TCEquation& desc, char mode)
{
    return std::find(desc.modeC.begin(), desc.modeC.end(), mode) != desc.modeC.end();
}

bool compute_gemm_transpose_for_left(
    const TconT::TCEquation& desc,
    const std::vector<char>& input_modes,
    const std::vector<int>& permutation)
{
    validate_permutation(permutation, input_modes.size(), "left input");
    return !is_external_mode(desc, input_modes[static_cast<size_t>(permutation[0])]);
}

bool compute_gemm_transpose_for_right(
    const TconT::TCEquation& desc,
    const std::vector<char>& input_modes,
    const std::vector<int>& permutation)
{
    validate_permutation(permutation, input_modes.size(), "right input");
    return is_external_mode(desc, input_modes[static_cast<size_t>(permutation[0])]);
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

template <typename ValueType>
void initialize_input_tensors(ValueType* left, int size_left, ValueType* right, int size_right)
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    for (int i = 0; i < size_left; ++i) {
        left[i] = static_cast<ValueType>(static_cast<double>(std::rand()) / RAND_MAX);
    }
    for (int i = 0; i < size_right; ++i) {
        right[i] = static_cast<ValueType>(static_cast<double>(std::rand()) / RAND_MAX);
    }
}

tc build_tc_from_equation(const TconT::TCEquation& desc)
{
    tc info{};
    info.len_output = static_cast<int>(desc.modeC.size());
    info.len_input_left = static_cast<int>(desc.modeA.size());
    info.len_input_right = static_cast<int>(desc.modeB.size());
    info.info_output = static_cast<idx*>(std::calloc(info.len_output, sizeof(idx)));
    info.info_input_left = static_cast<idx*>(std::calloc(info.len_input_left, sizeof(idx)));
    info.info_input_right = static_cast<idx*>(std::calloc(info.len_input_right, sizeof(idx)));

    if (info.info_output == nullptr || info.info_input_left == nullptr || info.info_input_right == nullptr) {
        throw std::bad_alloc();
    }

    auto fill_tensor = [&](idx* dst, int len, const std::vector<char>& modes, int tensor_type) {
        int size = 1;
        for (int i = 0; i < len; ++i) {
            const char mode = modes[static_cast<size_t>(i)];
            const auto it = desc.extent.find(mode);
            if (it == desc.extent.end()) {
                throw std::invalid_argument(std::string("Missing extent for mode: ") + mode);
            }

            dst[i].name[0] = mode;
            dst[i].name[1] = '\0';
            dst[i].size = static_cast<int>(it->second);
            dst[i].tile_size = static_cast<int>(it->second);
            dst[i].idx_type = (std::find(desc.modeC.begin(), desc.modeC.end(), mode) != desc.modeC.end()) ?
                TYPE_EXTERNAL : TYPE_INTERNAL;
            size *= dst[i].size;
        }
        return size;
    };

    info.size_output = fill_tensor(info.info_output, info.len_output, desc.modeC, TYPE_TENSOR_C);
    info.size_input_left = fill_tensor(info.info_input_left, info.len_input_left, desc.modeA, TYPE_TENSOR_A);
    info.size_input_right = fill_tensor(info.info_input_right, info.len_input_right, desc.modeB, TYPE_TENSOR_B);

    std::vector<char> internal_modes;
    for (char mode : desc.modeA) {
        if (std::find(desc.modeC.begin(), desc.modeC.end(), mode) == desc.modeC.end()) {
            internal_modes.push_back(mode);
        }
    }

    info.len_internal_indices = static_cast<int>(internal_modes.size());
    info.info_internal_indices = static_cast<idx*>(std::calloc(info.len_internal_indices, sizeof(idx)));
    if (info.info_internal_indices == nullptr && info.len_internal_indices > 0) {
        throw std::bad_alloc();
    }

    for (int i = 0; i < info.len_internal_indices; ++i) {
        const char mode = internal_modes[static_cast<size_t>(i)];
        const auto it = desc.extent.find(mode);
        info.info_internal_indices[i].name[0] = mode;
        info.info_internal_indices[i].name[1] = '\0';
        info.info_internal_indices[i].size = static_cast<int>(it->second);
        info.info_internal_indices[i].tile_size = static_cast<int>(it->second);
        info.info_internal_indices[i].idx_type = TYPE_INTERNAL;
    }

    info.op = (desc.op == '-') ? 2 : 1;
    return info;
}

void destroy_tc(tc& info)
{
    std::free(info.info_output);
    std::free(info.info_input_left);
    std::free(info.info_input_right);
    std::free(info.info_internal_indices);
    info.info_output = nullptr;
    info.info_input_left = nullptr;
    info.info_input_right = nullptr;
    info.info_internal_indices = nullptr;
}

void destroy_tt(tt* transpose)
{
    if (transpose == nullptr) {
        return;
    }
    std::free(transpose->ttlg_dims);
    std::free(transpose->ttlg_perms);
    std::free(transpose);
}

void destroy_tiles(tiles* tile_info)
{
    if (tile_info == nullptr) {
        return;
    }
    std::free(tile_info->info_slice);
    std::free(tile_info);
}

class TTGTRunImpl final : public TconT::RunImpl {
public:
    TTGTRunImpl(std::shared_ptr<const ttgt_cuTT_handle> plan, const TconT::TCEquation& desc)
        : plan_(std::move(plan))
    {
        if (desc.scalar_type != TconT::ScalarType::Float64) {
            throw std::invalid_argument("TTGT backend currently supports Float64 only");
        }

        const size_t elements_a = count_elements(desc.modeA, desc.extent);
        const size_t elements_b = count_elements(desc.modeB, desc.extent);
        const size_t elements_c = count_elements(desc.modeC, desc.extent);
        prepare_buffers<double>(elements_a, elements_b, elements_c);
        ttgt_cuTT_prepare(
            *plan_,
            runtime_,
            static_cast<double*>(device_a_raw_),
            static_cast<double*>(device_a_trans_raw_),
            static_cast<double*>(device_b_raw_),
            static_cast<double*>(device_b_trans_raw_),
            static_cast<double*>(device_c_raw_),
            static_cast<double*>(device_c_trans_raw_));
        device_output_raw_ = plan_->transpose_output ? device_c_raw_ : device_c_trans_raw_;
    }

    ~TTGTRunImpl() override
    {
        try {
            ttgt_cuTT_runtime_destroy(runtime_);
        } catch (...) {
        }
    }

    TconT::Backend backend() const override
    {
        return TconT::Backend::TTGT_CUTT;
    }

    void launch() override
    {
        double* output_ptr = static_cast<double*>(device_c_raw_);
        ttgt_cuTT_execute(
            *plan_,
            runtime_,
            static_cast<double*>(device_a_raw_),
            static_cast<double*>(device_a_trans_raw_),
            static_cast<double*>(device_b_raw_),
            static_cast<double*>(device_b_trans_raw_),
            output_ptr,
            static_cast<double*>(device_c_trans_raw_));
        device_output_raw_ = output_ptr;
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
        HANDLE_CUDA_ERROR(cudaMemcpy(destination, device_output_raw_, bytes, cudaMemcpyDeviceToHost));
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

        ValueType* host_a = reinterpret_cast<ValueType*>(host_a_storage_.data());
        ValueType* host_b = reinterpret_cast<ValueType*>(host_b_storage_.data());
        initialize_input_tensors(
            host_a,
            static_cast<int>(elements_a),
            host_b,
            static_cast<int>(elements_b));

        void* dA = nullptr;
        void* dATrans = nullptr;
        void* dB = nullptr;
        void* dBTrans = nullptr;
        void* dC = nullptr;
        void* dCTrans = nullptr;

        HANDLE_CUDA_ERROR(cudaMalloc(&dA, sizeof(ValueType) * elements_a));
        HANDLE_CUDA_ERROR(cudaMalloc(&dATrans, sizeof(ValueType) * elements_a));
        HANDLE_CUDA_ERROR(cudaMalloc(&dB, sizeof(ValueType) * elements_b));
        HANDLE_CUDA_ERROR(cudaMalloc(&dBTrans, sizeof(ValueType) * elements_b));
        HANDLE_CUDA_ERROR(cudaMalloc(&dC, sizeof(ValueType) * elements_c));
        HANDLE_CUDA_ERROR(cudaMalloc(&dCTrans, sizeof(ValueType) * elements_c));

        device_a_.reset(dA);
        device_a_trans_.reset(dATrans);
        device_b_.reset(dB);
        device_b_trans_.reset(dBTrans);
        device_c_.reset(dC);
        device_c_trans_.reset(dCTrans);

        device_a_raw_ = device_a_.get();
        device_a_trans_raw_ = device_a_trans_.get();
        device_b_raw_ = device_b_.get();
        device_b_trans_raw_ = device_b_trans_.get();
        device_c_raw_ = device_c_.get();
        device_c_trans_raw_ = device_c_trans_.get();
        device_output_raw_ = device_c_raw_;

        HANDLE_CUDA_ERROR(cudaMemcpy(device_a_raw_, host_a, sizeof(ValueType) * elements_a, cudaMemcpyHostToDevice));
        HANDLE_CUDA_ERROR(cudaMemcpy(device_b_raw_, host_b, sizeof(ValueType) * elements_b, cudaMemcpyHostToDevice));
        HANDLE_CUDA_ERROR(cudaMemset(device_c_raw_, 0, sizeof(ValueType) * elements_c));
        HANDLE_CUDA_ERROR(cudaMemset(device_c_trans_raw_, 0, sizeof(ValueType) * elements_c));
    }

    std::shared_ptr<const ttgt_cuTT_handle> plan_;
    ttgt_cuTT_runtime runtime_;
    std::unique_ptr<void, DeviceDeleter> device_a_;
    std::unique_ptr<void, DeviceDeleter> device_a_trans_;
    std::unique_ptr<void, DeviceDeleter> device_b_;
    std::unique_ptr<void, DeviceDeleter> device_b_trans_;
    std::unique_ptr<void, DeviceDeleter> device_c_;
    std::unique_ptr<void, DeviceDeleter> device_c_trans_;
    void* device_a_raw_ = nullptr;
    void* device_a_trans_raw_ = nullptr;
    void* device_b_raw_ = nullptr;
    void* device_b_trans_raw_ = nullptr;
    void* device_c_raw_ = nullptr;
    void* device_c_trans_raw_ = nullptr;
    void* device_output_raw_ = nullptr;
    std::vector<unsigned char> host_a_storage_;
    std::vector<unsigned char> host_b_storage_;
};

class TTGTPlanImpl final : public TconT::PlanImpl {
public:
    TTGTPlanImpl(ttgt_cuTT_handle plan, TconT::TCEquation equation)
        : plan_(std::make_shared<ttgt_cuTT_handle>(std::move(plan))),
          equation_(std::move(equation)) {}

    ~TTGTPlanImpl() override
    {
        ttgt_cuTT_plan_destroy(*plan_);
    }

    TconT::Backend backend() const override
    {
        return TconT::Backend::TTGT_CUTT;
    }

    std::shared_ptr<TconT::RunImpl> prepare() const override
    {
        return std::make_shared<TTGTRunImpl>(plan_, equation_);
    }

private:
    std::shared_ptr<ttgt_cuTT_handle> plan_;
    TconT::TCEquation equation_;
};
}  // namespace

std::shared_ptr<TconT::PlanImpl> plan_ttgt_cutt(const TconT::TCEquation& desc)
{
    if (desc.scalar_type != TconT::ScalarType::Float64) {
        throw std::invalid_argument("TTGT backend currently supports Float64 only");
    }

    check_cuda_status(cudaFree(nullptr), "cudaFree(nullptr)");

    ttgt_cuTT_handle plan;
    tc info_tc = build_tc_from_equation(desc);
    tiles* result_tiles = nullptr;
    tt* result_tt_output = nullptr;
    tt* result_tt_input_left = nullptr;
    tt* result_tt_input_right = nullptr;

    try {
        const int manual_config = -1;
        const int result_type = ttgt_enumeration(
            &info_tc,
            result_tiles,
            result_tt_output,
            result_tt_input_left,
            result_tt_input_right,
            manual_config,
            manual_config);

        plan.swap_AB = (result_type == TYPE_BA);
        plan.info_trans_C_perm.assign(
            result_tt_output->ttlg_perms,
            result_tt_output->ttlg_perms + result_tt_output->len_indice);
        plan.info_trans_A_perm.assign(
            result_tt_input_left->ttlg_perms,
            result_tt_input_left->ttlg_perms + result_tt_input_left->len_indice);
        plan.info_trans_B_perm.assign(
            result_tt_input_right->ttlg_perms,
            result_tt_input_right->ttlg_perms + result_tt_input_right->len_indice);

        const std::vector<char>& left_modes = plan.swap_AB ? desc.modeB : desc.modeA;
        const std::vector<char>& right_modes = plan.swap_AB ? desc.modeA : desc.modeB;

        validate_permutation(plan.info_trans_A_perm, left_modes.size(), "A");
        validate_permutation(plan.info_trans_B_perm, right_modes.size(), "B");

        fill_dims_from_modes(plan.info_trans_A_dim, left_modes, desc.extent);
        fill_dims_from_modes(plan.info_trans_B_dim, right_modes, desc.extent);
        plan.info_trans_C_dim = build_transposed_input_dims(
            desc.modeC,
            desc.extent,
            plan.info_trans_C_perm);

        plan.transpose_input_left = !is_identity_permutation(plan.info_trans_A_perm);
        plan.transpose_input_right = !is_identity_permutation(plan.info_trans_B_perm);
        plan.transpose_output = !is_identity_permutation(plan.info_trans_C_perm);

        if (plan.swap_AB) {
            plan.gemm_m = result_tiles->dgemm_n;
            plan.gemm_n = result_tiles->dgemm_m;
        } else {
            plan.gemm_m = result_tiles->dgemm_m;
            plan.gemm_n = result_tiles->dgemm_n;
        }
        plan.gemm_k = result_tiles->dgemm_k;
        plan.gemm_trans_A = compute_gemm_transpose_for_left(desc, left_modes, plan.info_trans_A_perm);
        plan.gemm_trans_B = compute_gemm_transpose_for_right(desc, right_modes, plan.info_trans_B_perm);
    } catch (...) {
        destroy_tt(result_tt_output);
        destroy_tt(result_tt_input_left);
        destroy_tt(result_tt_input_right);
        destroy_tiles(result_tiles);
        destroy_tc(info_tc);
        throw;
    }

    destroy_tt(result_tt_output);
    destroy_tt(result_tt_input_left);
    destroy_tt(result_tt_input_right);
    destroy_tiles(result_tiles);
    destroy_tc(info_tc);

    return std::make_shared<TTGTPlanImpl>(std::move(plan), desc);
}
}  // namespace ttgt_cutt_backend
