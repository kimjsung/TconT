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

void check_cublas_status(cublasStatus_t status, const char* expr)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS error in ") + expr);
    }
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

void print_ttgt_plan_debug(const ttgt_cuTT_handle& plan, const TconT::TCEquation& desc)
{
    (void)plan;
    (void)desc;
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

class TTGTRunImpl final : public TconT::RunImpl {
public:
    TTGTRunImpl(std::shared_ptr<ttgt_cuTT_handle> plan, const TconT::TCEquation& desc)
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
        device_output_raw_ = device_c_raw_;
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
            output_ptr,
            static_cast<double*>(device_c_trans_raw_));
        device_output_raw_ = output_ptr;
    }

    bool warmup(int iterations) override
    {
        if (iterations <= 0) {
            return true;
        }

        double* gemm_a = plan_->transpose_input_left
            ? static_cast<double*>(runtime_.exec_A_trans)
            : static_cast<double*>(runtime_.exec_A);
        double* gemm_b = plan_->transpose_input_right
            ? static_cast<double*>(runtime_.exec_B_trans)
            : static_cast<double*>(runtime_.exec_B);
        double* gemm_c = static_cast<double*>(device_c_trans_raw_);

        const double alpha = 1.0;
        const double beta = 0.0;
        const cublasOperation_t op_a = plan_->gemm_trans_A ? CUBLAS_OP_T : CUBLAS_OP_N;
        const cublasOperation_t op_b = plan_->gemm_trans_B ? CUBLAS_OP_T : CUBLAS_OP_N;
        const int lda = plan_->gemm_trans_A ? plan_->gemm_k : plan_->gemm_m;
        const int ldb = plan_->gemm_trans_B ? plan_->gemm_n : plan_->gemm_k;

        for (int i = 0; i < iterations; ++i) {
            check_cublas_status(
                cublasGemmEx(
                    runtime_.cublas_handle,
                    op_a,
                    op_b,
                    plan_->gemm_m,
                    plan_->gemm_n,
                    plan_->gemm_k,
                    &alpha,
                    gemm_a,
                    CUDA_R_64F,
                    lda,
                    gemm_b,
                    CUDA_R_64F,
                    ldb,
                    &beta,
                    gemm_c,
                    CUDA_R_64F,
                    plan_->gemm_m,
                    CUBLAS_COMPUTE_64F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                "cublasGemmEx(warmup)");
        }

        check_cuda_status(cudaDeviceSynchronize(), "cudaDeviceSynchronize(ttgt warmup)");
        return true;
    }

    void zero_output() override
    {
        HANDLE_CUDA_ERROR(cudaMemset(device_c_raw_, 0, output_bytes_));
        HANDLE_CUDA_ERROR(cudaMemset(device_c_trans_raw_, 0, output_bytes_));
        device_output_raw_ = device_c_raw_;
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
        output_bytes_ = sizeof(ValueType) * elements_c;

        HANDLE_CUDA_ERROR(cudaMemcpy(device_a_raw_, host_a, sizeof(ValueType) * elements_a, cudaMemcpyHostToDevice));
        HANDLE_CUDA_ERROR(cudaMemcpy(device_b_raw_, host_b, sizeof(ValueType) * elements_b, cudaMemcpyHostToDevice));
        HANDLE_CUDA_ERROR(cudaMemset(device_c_raw_, 0, sizeof(ValueType) * elements_c));
        HANDLE_CUDA_ERROR(cudaMemset(device_c_trans_raw_, 0, sizeof(ValueType) * elements_c));
    }

    std::shared_ptr<ttgt_cuTT_handle> plan_;
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
    size_t output_bytes_ = 0;
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
    ttgt_cuTT_plan(plan, desc.modeC, desc.modeA, desc.modeB, desc.extent, -1);
    print_ttgt_plan_debug(plan, desc);

    return std::make_shared<TTGTPlanImpl>(std::move(plan), desc);
}
}  // namespace ttgt_cutt_backend
