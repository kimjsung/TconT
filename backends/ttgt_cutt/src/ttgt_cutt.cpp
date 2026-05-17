/**
 * @file ttgt_cutt.cpp
 * @author Jinsung Kim (kimjsung@cau.ac.kr)
 * @brief
 *          (1) TTGT (Transpose-Transpose-GEMM-Transpose) for Tensor Contractions
 *          (2) A Tensor Contraction based on cuTT and cuBLAS
 * @version 0.1
 * @date 2026-02-07
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <getopt.h>

#include <cuda_runtime.h>

#include <cmath>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <stdexcept>

#include "ttgt_cutt.hpp"
#include "ttgt_cutt_models.hpp"

#include <cutt.h>

#define cuttCheck(stmt) do { \
	cuttResult err = stmt; \
	if (err != CUTT_SUCCESS) { \
		fprintf(stderr, "%s in file %s, function %s\n", #stmt,__FILE__,__FUNCTION__); \
		exit(1); \
	} \
} while(0)

namespace
{
void check_cuda_or_throw(cudaError_t status, const char* expr)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error in ") + expr + ": " + cudaGetErrorString(status));
    }
}

void check_cublas_or_throw(cublasStatus_t status, const char* expr)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS error in ") + expr);
    }
}
}

unsigned long long helper_factorial(unsigned long long n) {
  unsigned long long result = 1;
  for (unsigned long long i = 1; i <= n; ++i) {
    result *= i;
  }
  return result;
}

void helper_perm(char* array, int n, int i, int* offset, char* output)
{
  if (i == n) {
    for (int j = 0; j < n; j++) {
      output[*offset * n + j] = array[j];
    }
    (*offset)++;
    return;
  }
  for (int j = i; j < n; j++) {
    std::swap(array[i], array[j]);
    helper_perm(array, n, i + 1, offset, output);
    std::swap(array[i], array[j]);
  }
}

inline void fill_dims_from_modes(
  std::vector<int>& dims,
  const std::vector<char>& modes,
  const std::unordered_map<char, int64_t>& extent)
{
  dims.clear();
  dims.reserve(modes.size());

  for (char mode : modes) {
    auto it = extent.find(mode);
    if (it == extent.end()) {
      throw std::runtime_error(std::string("Extent for index '") + mode + "' not found.");
    }
    dims.push_back(static_cast<int>(it->second));
  }
}

inline int64_t find_dim_from_idx(
  const char idx,
  const std::unordered_map<char, int64_t>& extent)
{
  auto it = extent.find(idx);
  if (it == extent.end()) {
    throw std::runtime_error(std::string("Extent for index '") + idx + "' not found.");
  }
  return it->second;
}

void tmp_check_correctness_comparison(int total_size, double* output_host, double* output_device)
{
    if (total_size < 0) {
        fprintf(stderr, "Invalid total size: %d\n", total_size);
        return;
    }
    if (!output_host || !output_device) {
        fprintf(stderr, "Null pointer detected for output arrays.\n");
        return;
    }
    printf ("===========================================================================\n");

    const double abs_tol = 1e-11;
    const double rel_tol = 1e-9;

    int diff = 0;
    int same = 0;
    int non_finite = 0;

    const int max_print = 4;

    for (int i = 0; i < total_size; i++)
    {
        const double h = output_host[i];
        const double d = output_device[i];

        if (!std::isfinite(h) || !std::isfinite(d))
        {
            non_finite++;
            diff++;
            if (diff <= max_print) {
                printf ("[%d/%d] Non-finite value: host=%f, device=%f\n", i, total_size, h, d);
            }
            continue;
        }

        const double err = std::fabs(h - d);
        const double scale = std::max(std::fabs(h), std::fabs(d));
        const double tol = std::max(abs_tol, rel_tol * scale);

        if (err > tol)
        {
            diff++;
            if (diff <= max_print)
            {
                printf ("[%d/%d] host=%0.15e, device=%0.15e, |err|=%.03e, tol=%0.3e\n",
                    i, total_size, h, d, err, tol);
            }
        } else {
            same++;
        }

    }
    printf ("Differences: %d / %d (Same: %d, Non-finite: %d)\n",
        diff, total_size, same, non_finite);
    printf ("===========================================================================\n");
}

inline void print_vector_info(const char* label, const std::vector<int>& values)
{
  printf("%s: [", label);
  for (size_t i = 0; i < values.size(); ++i) {
    printf("%d", values[i]);
    if (i + 1 < values.size()) {
      printf(", ");
    }
  }
  printf("]\n");
}

void ttgt_cuTT_plan(
  ttgt_cuTT_handle& plan,
  const std::vector<char>& modeC,
  const std::vector<char>& modeA,
  const std::vector<char>& modeB,
  const std::unordered_map<char, int64_t>& extent,
  unsigned int target_config)
{
  std::vector<char> idx_ext_left;
  std::vector<char> idx_ext_right;
  std::vector<char> idx_int;

  unsigned int len_gemm_m = 1;
  unsigned int len_gemm_n = 1;
  unsigned int len_gemm_k = 1;

  for (char m : modeA) {
    if (std::find(modeC.begin(), modeC.end(), m) != modeC.end()) {
      idx_ext_left.push_back(m);
      len_gemm_m *= find_dim_from_idx(m, extent);
    } else {
      idx_int.push_back(m);
      len_gemm_k *= find_dim_from_idx(m, extent);
    }
  }

  for (char m : modeB) {
    if (std::find(modeC.begin(), modeC.end(), m) != modeC.end()) {
      idx_ext_right.push_back(m);
      len_gemm_n *= find_dim_from_idx(m, extent);
    }
  }

  size_t size_A = len_gemm_m * len_gemm_k * sizeof(double);
  size_t size_B = len_gemm_k * len_gemm_n * sizeof(double);
  size_t size_C = len_gemm_m * len_gemm_n * sizeof(double);

  printf ("[%s] A: %lu bytes, B: %lu bytes, C: %lu bytes\n", __func__, size_A, size_B, size_C);

  int device_id;
  cudaGetDevice(&device_id);

  int rtx = true;
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id);

  int num_fp32_units_per_sm = 0;
  if (major == 7 && minor == 0) num_fp32_units_per_sm = 64;
  if (major == 7 && minor == 5) num_fp32_units_per_sm = 64;
  if (major == 8 && minor == 0) num_fp32_units_per_sm = 64;
  if (major == 8 && minor == 6) num_fp32_units_per_sm = 128;
  if (major == 8 && minor == 9) num_fp32_units_per_sm = 128;
  if (major == 9 && minor == 0) num_fp32_units_per_sm = 128;

  int num_sms = 0;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id);

  double memory_clock_khz = 0.0;
  int bus_width = 0;
  cudaDeviceGetAttribute((int*)&memory_clock_khz, cudaDevAttrMemoryClockRate, device_id);
  cudaDeviceGetAttribute(&bus_width, cudaDevAttrGlobalMemoryBusWidth, device_id);

  double peak_bw = 2.0 * memory_clock_khz * 1000.0 * (bus_width / 8.0) / 1.0e9;
  printf ("[%s] Peak Memory Bandwidth: %.2f GB/s\n", __func__, peak_bw);

  double min_trans_A_time_ms = (2.0 * size_A / 1.0e9) / peak_bw * 1000.0;
  double min_trans_B_time_ms = (2.0 * size_B / 1.0e9) / peak_bw * 1000.0;
  double min_trans_C_time_ms = (2.0 * size_C / 1.0e9) / peak_bw * 1000.0;
  printf ("[%s] Estimated time for transposing A: %.2f ms\n", __func__, min_trans_A_time_ms);
  printf ("[%s] Estimated time for transposing B: %.2f ms\n", __func__, min_trans_B_time_ms);
  printf ("[%s] Estimated time for transposing C: %.2f ms\n", __func__, min_trans_C_time_ms);

  int fp64_per_sm = getFP64CoresPerSM(major, minor);
  double peak_fp64_tflops = static_cast<double>(num_sms * fp64_per_sm) * 2.0 * 1.0 / 1.0e3;
  printf ("[%s] Peak FP64 FLOPS: %.2f TFLOPS\n", __func__, peak_fp64_tflops);

  int num_trans_perms_swapped = 8;
  std::vector<new_config> list_configs(num_trans_perms_swapped);

  list_configs[0].swap = false;
  list_configs[1].swap = false;
  list_configs[2].swap = false;
  list_configs[3].swap = false;
  list_configs[4].swap = true;
  list_configs[5].swap = true;
  list_configs[6].swap = true;
  list_configs[7].swap = true;

  for (int i = 0; i < num_trans_perms_swapped; i++) {
    list_configs[i].useless = true;
  }

  unsigned int model_result = ttgt_cuTT_model_overall_analytic(
    list_configs.data(),
    num_trans_perms_swapped,
    min_trans_A_time_ms,
    min_trans_B_time_ms,
    min_trans_C_time_ms,
    target_config);
  printf ("[%s] # Selected Configs: %u / %u\n", __func__, model_result, num_trans_perms_swapped);

  for (int i = 0; i < num_trans_perms_swapped; i++) {
    if (list_configs[i].useless == true) {
      printf ("[%s] Discarded Config #%d\n", __func__, i);
    } else {
      printf ("[%s] Kept Config #%d\n", __func__, i);
    }
  }

  plan.gemm_k = len_gemm_k;
  if (target_config < static_cast<unsigned int>(num_trans_perms_swapped))
  {
    plan.swap_AB = list_configs[target_config].swap;
    if (plan.swap_AB == true)
    {
      plan.gemm_m = len_gemm_n;
      plan.gemm_n = len_gemm_m;

      fill_dims_from_modes(plan.info_trans_A_dim, modeB, extent);
      fill_dims_from_modes(plan.info_trans_B_dim, modeA, extent);

      plan.transpose_input_left = list_configs[target_config].trans_input_right;
      plan.transpose_input_right = list_configs[target_config].trans_input_left;

      plan.info_trans_A_perm = list_configs[target_config].info_perms_input_right;
      plan.info_trans_B_perm = list_configs[target_config].info_perms_input_left;

      plan.gemm_trans_A = list_configs[target_config].gemm_trans_B;
      plan.gemm_trans_B = list_configs[target_config].gemm_trans_A;
    }
    else
    {
      plan.gemm_m = len_gemm_m;
      plan.gemm_n = len_gemm_n;

      fill_dims_from_modes(plan.info_trans_A_dim, modeA, extent);
      fill_dims_from_modes(plan.info_trans_B_dim, modeB, extent);

      plan.transpose_input_left = list_configs[target_config].trans_input_left;
      plan.transpose_input_right = list_configs[target_config].trans_input_right;

      plan.info_trans_A_perm = list_configs[target_config].info_perms_input_left;
      plan.info_trans_B_perm = list_configs[target_config].info_perms_input_right;

      plan.gemm_trans_A = list_configs[target_config].gemm_trans_A;
      plan.gemm_trans_B = list_configs[target_config].gemm_trans_B;
    }

    plan.transpose_output = list_configs[target_config].trans_output;
    plan.info_trans_C_perm = list_configs[target_config].info_perms_output;
    plan.info_trans_C_dim = list_configs[target_config].info_dims_output;
  }
  else
  {
    plan.swap_AB = list_configs[0].swap;
    if (plan.swap_AB == true)
    {
      plan.gemm_m = len_gemm_n;
      plan.gemm_n = len_gemm_m;

      fill_dims_from_modes(plan.info_trans_A_dim, modeB, extent);
      fill_dims_from_modes(plan.info_trans_B_dim, modeA, extent);

      plan.transpose_input_left = list_configs[0].trans_input_right;
      plan.transpose_input_right = list_configs[0].trans_input_left;

      plan.info_trans_A_perm = list_configs[0].info_perms_input_right;
      plan.info_trans_B_perm = list_configs[0].info_perms_input_left;

      plan.gemm_trans_A = list_configs[0].gemm_trans_B;
      plan.gemm_trans_B = list_configs[0].gemm_trans_A;
    }
    else
    {
      plan.gemm_m = len_gemm_m;
      plan.gemm_n = len_gemm_n;

      fill_dims_from_modes(plan.info_trans_A_dim, modeA, extent);
      fill_dims_from_modes(plan.info_trans_B_dim, modeB, extent);

      plan.transpose_input_left = list_configs[0].trans_input_left;
      plan.transpose_input_right = list_configs[0].trans_input_right;

      plan.info_trans_A_perm = list_configs[0].info_perms_input_left;
      plan.info_trans_B_perm = list_configs[0].info_perms_input_right;

      plan.gemm_trans_A = list_configs[0].gemm_trans_A;
      plan.gemm_trans_B = list_configs[0].gemm_trans_B;
    }

    plan.transpose_output = list_configs[0].trans_output;
    plan.info_trans_C_perm = list_configs[0].info_perms_output;
    plan.info_trans_C_dim = list_configs[0].info_dims_output;
  }
}

template <typename T>
void ttgt_cuTT_prepare(
  const ttgt_cuTT_handle& plan,
  ttgt_cuTT_runtime& runtime,
  T* d_A, T* d_A_trans, T* d_B, T* d_B_trans, T* d_C, T* d_C_trans)
{
  check_cublas_or_throw(cublasCreate(&runtime.cublas_handle), "cublasCreate");

  T* exec_A = d_A;
  T* exec_A_trans = d_A_trans;
  T* exec_B = d_B;
  T* exec_B_trans = d_B_trans;
  if (plan.swap_AB) {
    std::swap(exec_A, exec_B);
    std::swap(exec_A_trans, exec_B_trans);
  }

  if (plan.transpose_input_left) {
    cuttCheck(cuttPlanMeasure(
      &runtime.cutt_handle_A,
      static_cast<int>(plan.info_trans_A_dim.size()),
      const_cast<int*>(plan.info_trans_A_dim.data()),
      const_cast<int*>(plan.info_trans_A_perm.data()),
      sizeof(T),
      0,
      exec_A,
      exec_A_trans));
    runtime.has_cutt_handle_A = true;
  }

  if (plan.transpose_input_right) {
    cuttCheck(cuttPlanMeasure(
      &runtime.cutt_handle_B,
      static_cast<int>(plan.info_trans_B_dim.size()),
      const_cast<int*>(plan.info_trans_B_dim.data()),
      const_cast<int*>(plan.info_trans_B_perm.data()),
      sizeof(T),
      0,
      exec_B,
      exec_B_trans));
    runtime.has_cutt_handle_B = true;
  }

  if (plan.transpose_output) {
    cuttCheck(cuttPlanMeasure(
      &runtime.cutt_handle_C,
      static_cast<int>(plan.info_trans_C_dim.size()),
      const_cast<int*>(plan.info_trans_C_dim.data()),
      const_cast<int*>(plan.info_trans_C_perm.data()),
      sizeof(T),
      0,
      d_C,
      d_C_trans));
    runtime.has_cutt_handle_C = true;
  }
}

template <typename T>
void ttgt_cuTT_execute(const ttgt_cuTT_handle& plan,
  ttgt_cuTT_runtime& runtime,
  T* d_A, T* d_A_trans, T* d_B, T* d_B_trans, T*& d_C, T* d_C_trans)
{
  T* exec_A = d_A;
  T* exec_A_trans = d_A_trans;
  T* exec_B = d_B;
  T* exec_B_trans = d_B_trans;

  if (plan.swap_AB) {
    std::swap(exec_A, exec_B);
    std::swap(exec_A_trans, exec_B_trans);
  }

  if (plan.transpose_input_left)
  {
    cuttCheck(cuttExecute(runtime.cutt_handle_A, exec_A, exec_A_trans));
  } else { exec_A_trans = exec_A; }

  if (plan.transpose_input_right)
  {
    cuttCheck(cuttExecute(runtime.cutt_handle_B, exec_B, exec_B_trans));
  } else { exec_B_trans = exec_B; }

  const double gemm_alpha = 1.0;
  const double gemm_beta = 0.0;

  if (plan.gemm_trans_A == false)
  {
    if (plan.gemm_trans_B == false)
    {
      check_cublas_or_throw(cublasGemmEx(runtime.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                plan.gemm_m, plan.gemm_n, plan.gemm_k,
                                &gemm_alpha,  exec_A_trans, CUDA_R_64F, plan.gemm_m,
                                              exec_B_trans, CUDA_R_64F, plan.gemm_k,
                                              &gemm_beta,
                                              d_C_trans, CUDA_R_64F, plan.gemm_m,
                                              CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "cublasGemmEx(N,N)");
    }
    else
    {
      check_cublas_or_throw(cublasGemmEx(runtime.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                plan.gemm_m, plan.gemm_n, plan.gemm_k,
                                &gemm_alpha,  exec_A_trans, CUDA_R_64F, plan.gemm_m,
                                              exec_B_trans, CUDA_R_64F, plan.gemm_n,
                                              &gemm_beta,
                                              d_C_trans, CUDA_R_64F, plan.gemm_m,
                                              CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "cublasGemmEx(N,T)");
    }
  }
  else
  {
    if (plan.gemm_trans_B == false)
    {
      check_cublas_or_throw(cublasGemmEx(runtime.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                plan.gemm_m, plan.gemm_n, plan.gemm_k,
                                &gemm_alpha,  exec_A_trans, CUDA_R_64F, plan.gemm_k,
                                              exec_B_trans, CUDA_R_64F, plan.gemm_k,
                                              &gemm_beta,
                                              d_C_trans, CUDA_R_64F, plan.gemm_m,
                                              CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "cublasGemmEx(T,N)");
    }
    else
    {
      check_cublas_or_throw(cublasGemmEx(runtime.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T,
                                plan.gemm_m, plan.gemm_n, plan.gemm_k,
                                &gemm_alpha,  exec_A_trans, CUDA_R_64F, plan.gemm_k,
                                              exec_B_trans, CUDA_R_64F, plan.gemm_n,
                                              &gemm_beta,
                                              d_C_trans, CUDA_R_64F, plan.gemm_m,
                                              CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "cublasGemmEx(T,T)");
    }
  }

  if (plan.transpose_output)
  {
    cuttCheck(cuttExecute(runtime.cutt_handle_C, d_C_trans, d_C));
  }
  else { d_C = d_C_trans; }
}

void ttgt_cuTT_plan_destroy(ttgt_cuTT_handle& plan)
{
}

void ttgt_cuTT_runtime_destroy(ttgt_cuTT_runtime& runtime)
{
  if (runtime.has_cutt_handle_A) {
    cuttDestroy(runtime.cutt_handle_A);
    runtime.has_cutt_handle_A = false;
  }
  if (runtime.has_cutt_handle_B) {
    cuttDestroy(runtime.cutt_handle_B);
    runtime.has_cutt_handle_B = false;
  }
  if (runtime.has_cutt_handle_C) {
    cuttDestroy(runtime.cutt_handle_C);
    runtime.has_cutt_handle_C = false;
  }
  if (runtime.cublas_handle != nullptr) {
    cublasDestroy(runtime.cublas_handle);
    runtime.cublas_handle = nullptr;
  }
}

template void ttgt_cuTT_prepare<double>(const ttgt_cuTT_handle& plan, ttgt_cuTT_runtime& runtime, double* d_A, double* d_A_trans, double* d_B, double* d_B_trans, double* d_C, double* d_C_trans);
template void ttgt_cuTT_execute<double>(const ttgt_cuTT_handle& plan, ttgt_cuTT_runtime& runtime, double* d_A, double* d_A_trans, double* d_B, double* d_B_trans, double*& d_C, double* d_C_trans);
