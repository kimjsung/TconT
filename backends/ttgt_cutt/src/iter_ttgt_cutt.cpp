/**
 * @file ttgt_cutt.cpp
 * @author Jinsung Kim (kimjsung@cau.ac.kr)
 * @brief 
 *          (1) TTGT (Transpose-Transpose-GEMM-Transpose) for Tensor Contractions
 *          (2) A Tensor Contraction based on cuTT and cuBLAS
 *          (3) TCCG Benchmarking (outside)
 * 
 * @version 0.1
 * @date 2026-02-07
 * 
 * @copyright Copyright (c) 2026
 * 
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <getopt.h>

#include <cuda_runtime.h>
// #include <cutensor.h>

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
void check_cublas_or_throw(cublasStatus_t status, const char* expr)
{
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("cuBLAS error in ") + expr);
  }
}

void check_cuda_or_throw(cudaError_t status, const char* expr)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error in ") + expr + ": " + cudaGetErrorString(status));
  }
}
}

/**
 * @brief Helper function to calculate factorial of a number
 * 
 * @param n 
 * @return unsigned long long 
 */
unsigned long long helper_factorial(unsigned long long n) {
  unsigned long long result = 1;
  for (unsigned long long i = 1; i <= n; ++i) {
    result *= i;
  }
  return result;
}

/**
 * @brief Recursive helper function to generate permutations of an array
 * 
 * @param array 
 * @param n 
 * @param i 
 * @param offset 
 * @param output 
 */
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

/**
 * @brief 
 * 
 * @param dims 
 * @param modes 
 * @param extent 
 */
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

//
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
    
    // Tolerances:
    //  - abs_tol handles values near 0
    //  - rel_tol scales with magnitude for larger values
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

        // Handle NaN/Inf explicitly
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

/**
 * @brief 
 * 
 * @param label 
 * @param values 
 */
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

/**
 * @brief After the function, 
 *        we can run ttgt_cuTT_execution()
 * 
 */
void ttgt_cuTT_plan(
  ttgt_cuTT_handle& plan,
  const std::vector<char>& modeC, 
  const std::vector<char>& modeA, 
  const std::vector<char>& modeB, 
  const std::unordered_map<char, int64_t>& extent, 
  int target_config)
{
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // 
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

  int device_id;
  cudaGetDevice(&device_id);

  int rtx = true;
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id);

  int num_fp32_units_per_sm = 0;
  // Throughput-equivalent values for simple peak-FLOPS estimation
  if (major == 7 && minor == 0) num_fp32_units_per_sm = 64;   // Volta
  if (major == 7 && minor == 5) num_fp32_units_per_sm = 64;   // Turing

  // Practical baseline for the simple formula
  if (major == 8 && minor == 0) num_fp32_units_per_sm = 64;   // Ampere SM80 baseline
  if (major == 8 && minor == 6) num_fp32_units_per_sm = 128;  // 2x FP32 throughput vs 8.0
  if (major == 8 && minor == 9) num_fp32_units_per_sm = 128;  // 2x FP32 throughput vs 8.0
  if (major == 9 && minor == 0) num_fp32_units_per_sm = 128;  // 2x FP32 throughput vs 8.0

  /**
   * @brief analytical model for estimating the time of transpositions in TTGT
     - We can use the peak memory bandwidth of the GPU to estimate the time taken for transposing A, B, and C.
     - The time for transposing can be estimated as: Time = (2 * size of tensor) / (peak memory bandwidth)
       - The factor of 2 accounts for reading and writing during the transpose operation.
   * 
   */
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id);
  
  int core_clock_khz = 0;
  int memory_clock_khz = 0;
  int memory_bus_width_bits = 0;

  cudaDeviceGetAttribute(&core_clock_khz, cudaDevAttrClockRate, device_id);
  cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, device_id);
  cudaDeviceGetAttribute(&memory_bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, device_id);

  // Get SM Count
  int sm_count = prop.multiProcessorCount;
  double core_clock_ghz = core_clock_khz / 1.0e6;

  const double peak_bw_gbps =
      2.0 * static_cast<double>(memory_clock_khz) * 1000.0 *
      (static_cast<double>(memory_bus_width_bits) / 8.0) / 1.0e9;

  // 
  float min_trans_A_time_ms = 2 * size_A / (peak_bw_gbps * 1.0e9) * 1.0e3; //
  float min_trans_B_time_ms = 2 * size_B / (peak_bw_gbps * 1.0e9) * 1.0e3; //
  float min_trans_C_time_ms = 2 * size_C / (peak_bw_gbps * 1.0e9) * 1.0e3; //

  // 
  int fp64CoresPerSM = getFP64CoresPerSM(prop.major, prop.minor);
  double clockRateHz = core_clock_khz * 1000.0; // Convert kHz to Hz
  double peakFP64_Flops = (double)sm_count * fp64CoresPerSM * clockRateHz * 2.0;
  double peakFP64_TFlops = peakFP64_Flops / 1.0e12;
  
  // 
  int num_ext_idx_left    = idx_ext_left.size();
  int num_ext_idx_right   = idx_ext_right.size();
  int num_int_idx         = idx_int.size();

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // 
  int num_idx_output      = num_ext_idx_left + num_ext_idx_right;
  int num_idx_input_left  = num_ext_idx_left + num_int_idx;
  int num_idx_input_right = num_ext_idx_right + num_int_idx;

  // 
  int num_perms_ext_idx_left  = helper_factorial(num_ext_idx_left);
  int num_perms_ext_idx_right = helper_factorial(num_ext_idx_right);
  int num_perms_int_idx       = helper_factorial(num_int_idx);

  // 
  char* idx_perms_ext_input_left   = new char[num_ext_idx_left * num_perms_ext_idx_left];
  char* idx_perms_ext_input_right  = new char[num_ext_idx_right * num_perms_ext_idx_right];
  char* idx_perms_int_input        = new char[num_int_idx * num_perms_int_idx];

  int offset_left = 0;
  helper_perm(idx_ext_left.data(), num_ext_idx_left, 0, &offset_left, idx_perms_ext_input_left);

  int offset_right = 0; 
  helper_perm(idx_ext_right.data(), num_ext_idx_right, 0, &offset_right, idx_perms_ext_input_right);

  int offset_int = 0;
  helper_perm(idx_int.data(), num_int_idx, 0, &offset_int, idx_perms_int_input);

  // printf ("--------------------------------------------------------------\n");
  // for (int i = 0; i < num_perms_ext_idx_left; i++) {
  //   printf ("Permutation of external indices in Left Tensor #%d: ", i);
  //   for (int j = 0; j < num_ext_idx_left; j++) {
  //     printf ("%c ", idx_perms_ext_input_left[i * num_ext_idx_left + j]);
  //   }
  //   printf ("\n");
  // }

  // printf ("--------------------------------------------------------------\n");
  // for (int i = 0; i < num_perms_ext_idx_right; i++) {
  //   printf ("Permutation of external indices in Right Tensor #%d: ", i);
  //   for (int j = 0; j < num_ext_idx_right; j++) {
  //     printf ("%c ", idx_perms_ext_input_right[i * num_ext_idx_right + j]);
  //   }
  //   printf ("\n");
  // }

  // printf ("--------------------------------------------------------------\n");
  // for (int i = 0; i < num_perms_int_idx; i++) {
  //   printf ("Permutation of internal indices #%d: ", i);
  //   for (int j = 0; j < num_int_idx; j++) {
  //     printf ("%c ", idx_perms_int_input[i * num_int_idx + j]);
  //   }
  //   printf ("\n");
  // }
  // printf ("--------------------------------------------------------------\n");

  // Calculate the number of configurations
  int num_trans_perms = num_perms_ext_idx_left  * 2 *   // A[ext.,int.] and A[int.,ext.]
                        num_perms_ext_idx_right * 2 *   // B[ext.,int.] and B[int.,ext.] 
                        num_perms_int_idx;

  int num_trans_perms_swapped = num_trans_perms * 2; // For swapping A and B
  // printf ("[%s] # configurations: %d\n", __func__, num_trans_perms_swapped);

  // 
  int result_output[num_idx_output][2];
  int result_output_swapped[num_idx_output][2];

  // 
  new_config* list_configs = new new_config[num_trans_perms_swapped];

  /**
   * @brief Generate all configurations for TTGT
   *        : 1. Permutations of internal indices in both Left and Right Tensors
   *        : 2. Permutations of external indices in Left Tensor
   *        : 3. Permutations of external indices in Right Tensor
   *        : 4. Swapping internal and external indices in Left Tensor  // [Ext.,Int.] <-> [Int.,Ext.]
   *        : 5. Swapping internal and external indices in Right Tensor // [Ext.,Int.] <-> [Int.,Ext.]
   *        : 6. Swapping A and B 
   */
  int count = 0;
  // 1. Permutations of internal indices: |Internal Indices|! 
  for (int idx_perm_int = 0; idx_perm_int < num_perms_int_idx; idx_perm_int++) 
  {
    // 2-1. Swapping internal and external indices in Left Tensor: 0: A[ext.,int.], 1: A[int.,ext.]
    for (int idx_perm_ext_int_left = 0; idx_perm_ext_int_left < 2; idx_perm_ext_int_left++) 
    {
      // 2-2. External Indices in Left Tensor: |External Indices in Left Tensor|!
      for (int idx_perm_ext_left = 0; idx_perm_ext_left < num_perms_ext_idx_left; idx_perm_ext_left++) 
      {
        std::vector<char> target_left;
        std::vector<int> target_left_perms;
        if (idx_perm_ext_int_left == 0) 
        {
          // A[ext.,int.]
          target_left.insert(target_left.end(), idx_perms_ext_input_left + idx_perm_ext_left * num_ext_idx_left, idx_perms_ext_input_left + (idx_perm_ext_left + 1) * num_ext_idx_left);
          target_left.insert(target_left.end(), idx_perms_int_input + idx_perm_int * num_int_idx, idx_perms_int_input + (idx_perm_int + 1) * num_int_idx);
        } 
        else 
        {
          // A[int.,ext.]
          target_left.insert(target_left.end(), idx_perms_int_input + idx_perm_int * num_int_idx, idx_perms_int_input + (idx_perm_int + 1) * num_int_idx);
          target_left.insert(target_left.end(), idx_perms_ext_input_left + idx_perm_ext_left * num_ext_idx_left, idx_perms_ext_input_left + (idx_perm_ext_left + 1) * num_ext_idx_left);
        }
        /* @result A' */
        bool trans_A = false;
        for (int i = 0; i < target_left.size(); i++) {
          for (int j = 0; j < modeA.size(); j++) {
            if (modeA[j] == target_left[i]) {
              // printf ("A[%d] = A'[%d] (%c)\n", j, i, modeA[j]);
              if (trans_A == false && i != j) { trans_A = true; }
              target_left_perms.push_back(j);
              break;
            }
          }
        }

        // 3-1. Swapping internal and external indices in Right Tensor: 0: B[ext.,int.], 1: B[int.,ext.]
        for (int idx_perm_ext_int_right = 0; idx_perm_ext_int_right < 2; idx_perm_ext_int_right++) 
        {
          // 3-2. External Indices in Right Tensor: |External Indices in Right Tensor|!
          for (int idx_perm_ext_right = 0; idx_perm_ext_right < num_perms_ext_idx_right; idx_perm_ext_right++) 
          {
            std::vector<char> target_right;
            std::vector<int> target_right_perms;
            if (idx_perm_ext_int_right == 0)
            {
              // B[ext.,int.]
              target_right.insert(target_right.end(), idx_perms_ext_input_right + idx_perm_ext_right * num_ext_idx_right, idx_perms_ext_input_right + (idx_perm_ext_right + 1) * num_ext_idx_right);
              target_right.insert(target_right.end(), idx_perms_int_input + idx_perm_int * num_int_idx, idx_perms_int_input + (idx_perm_int + 1) * num_int_idx);
            } 
            else  
            {
              // B[int.,ext.]
              target_right.insert(target_right.end(), idx_perms_int_input + idx_perm_int * num_int_idx, idx_perms_int_input + (idx_perm_int + 1) * num_int_idx);
              target_right.insert(target_right.end(), idx_perms_ext_input_right + idx_perm_ext_right * num_ext_idx_right, idx_perms_ext_input_right + (idx_perm_ext_right + 1) * num_ext_idx_right);
            }
            /* @result B' */
            bool trans_B = false;
            for (int i = 0; i < target_right.size(); i++) {
              for (int j = 0; j < modeB.size(); j++) {
                if (modeB[j] == target_right[i]) {
                  // printf ("B[%d] = B'[%d] (%c)\n", j, i, modeB[j]);
                  if (trans_B == false && i != j) { trans_B = true; }
                  target_right_perms.push_back(j);
                  break;
                }
              }
            }

            // 6. Swapping A and B: 0: No swap, 1: Swap A and B
            // for (int swap_AB = 0; swap_AB < 2; swap_AB++) 
            for (int swap_AB = 0; swap_AB < 2; swap_AB++) 
            {
              std::vector<char> result_output;
              std::vector<int> result_output_perms;
              std::vector<int> result_output_dims; 
              if (swap_AB == 0) 
              {
                // Result Output from GEMM with No swap
                result_output.insert(result_output.end(), idx_perms_ext_input_left + idx_perm_ext_left * num_ext_idx_left, idx_perms_ext_input_left + (idx_perm_ext_left + 1) * num_ext_idx_left);

                result_output.insert(result_output.end(), idx_perms_ext_input_right + idx_perm_ext_right * num_ext_idx_right, idx_perms_ext_input_right + (idx_perm_ext_right + 1) * num_ext_idx_right);

                for (int i = 0; i < result_output.size(); i++) {
                  result_output_dims.push_back(find_dim_from_idx(result_output[i], extent));
                }

                /* @result C' */
                bool trans_C = false;
                for (int i = 0; i < modeC.size(); i++) {
                  for (int j = 0; j < result_output.size(); j++) {
                    if (modeC[i] == result_output[j]) {
                      if (trans_C == false && i != j) { trans_C = true; }
                      result_output_perms.push_back(j);
                      break;
                    }
                  }
                }

                /**
                 * @brief TODO: adding a configuration to the list
                 * 
                 */
                list_configs[count].swap                    = false;
                list_configs[count].trans_input_left        = trans_A;
                list_configs[count].trans_input_right       = trans_B;
                list_configs[count].trans_output            = trans_C;
                list_configs[count].info_perms_input_left   = target_left_perms;
                list_configs[count].info_perms_input_right  = target_right_perms;
                list_configs[count].info_perms_output       = result_output_perms;
                list_configs[count].info_dims_output        = result_output_dims;

                // print_vector_info(" target_left_perms", target_left_perms);
                
                // A[int.,ext] // T
                if (idx_perm_ext_int_left == 1) { list_configs[count].gemm_trans_A = true; } 
            
                // B[ext.,int] // T
                if (idx_perm_ext_int_right == 0) { list_configs[count].gemm_trans_B = true; }
            
                count++;
              } 
              else 
              {
                // Result Output from GEMM with Swapped A and B
                result_output.insert(result_output.end(), idx_perms_ext_input_right + idx_perm_ext_right * num_ext_idx_right, idx_perms_ext_input_right + (idx_perm_ext_right + 1) * num_ext_idx_right);
                result_output.insert(result_output.end(), idx_perms_ext_input_left + idx_perm_ext_left * num_ext_idx_left, idx_perms_ext_input_left + (idx_perm_ext_left + 1) * num_ext_idx_left);

                for (int i = 0; i < result_output.size(); i++) {
                  result_output_dims.push_back(find_dim_from_idx(result_output[i], extent));
                }

                /* @result C' */
                bool trans_C = false;
                for (int i = 0; i < modeC.size(); i++) {
                  for (int j = 0; j < result_output.size(); j++) {
                    if (modeC[i] == result_output[j]) {
                      // printf ("C[%d] = C'[%d] (%c)\n", i, j, modeC[i]);
                      if (trans_C == false && i != j) { trans_C = true; }
                      result_output_perms.push_back(j);
                      break;
                    }
                  }
                }

                /**
                 * @brief TODO: adding a configuration to the list
                 * 
                 */
                list_configs[count].swap                    = true;
                list_configs[count].trans_input_left        = trans_A;
                list_configs[count].trans_input_right       = trans_B;
                list_configs[count].trans_output            = trans_C;
                list_configs[count].info_perms_input_left   = target_left_perms;
                list_configs[count].info_perms_input_right  = target_right_perms;
                list_configs[count].info_perms_output       = result_output_perms;
                list_configs[count].info_dims_output        = result_output_dims;

                // A[int.,ext] // T
                if (idx_perm_ext_int_left == 0) { list_configs[count].gemm_trans_A = true; }
                
                // B[ext.,int] // T
                if (idx_perm_ext_int_right == 1) { list_configs[count].gemm_trans_B = true; }
                
                count++;
              }
            }
          }
        }
      }
    }
  }

  /**
   * @brief All Configurations
   * 
   */
  unsigned int model_result = ttgt_cuTT_model_overall_analytic(list_configs, num_trans_perms_swapped, 
    min_trans_A_time_ms, min_trans_B_time_ms, min_trans_C_time_ms,
    target_config);
  (void)model_result;

  int selected_config = 0;
  if (target_config < static_cast<unsigned int>(num_trans_perms_swapped))
  {
    selected_config = static_cast<int>(target_config);
  }
  else
  {
    bool found_selected = false;
    for (int i = 0; i < num_trans_perms_swapped; ++i) {
      if (list_configs[i].useless == false) {
        selected_config = i;
        found_selected = true;
        break;
      }
    }
    if (found_selected == false) {
      selected_config = 0;
    }
  }
  // Common values
  plan.gemm_k = len_gemm_k;
  if (selected_config < num_trans_perms_swapped) 
  {
    plan.swap_AB                = list_configs[selected_config].swap;
    if (plan.swap_AB == true)
    {
      plan.gemm_m = len_gemm_n;
      plan.gemm_n = len_gemm_m;

      fill_dims_from_modes(plan.info_trans_A_dim, modeB, extent);
      fill_dims_from_modes(plan.info_trans_B_dim, modeA, extent);

      plan.transpose_input_left   = list_configs[selected_config].trans_input_right;
      plan.transpose_input_right  = list_configs[selected_config].trans_input_left;

      plan.info_trans_A_perm      = list_configs[selected_config].info_perms_input_right;
      plan.info_trans_B_perm      = list_configs[selected_config].info_perms_input_left;
      
      plan.gemm_trans_A           = list_configs[selected_config].gemm_trans_B;
      plan.gemm_trans_B           = list_configs[selected_config].gemm_trans_A;
    }
    // (plan.swap_AB == false)
    else 
    {
      plan.gemm_m = len_gemm_m;
      plan.gemm_n = len_gemm_n;
  
      fill_dims_from_modes(plan.info_trans_A_dim, modeA, extent);
      fill_dims_from_modes(plan.info_trans_B_dim, modeB, extent);
      
      plan.transpose_input_left   = list_configs[selected_config].trans_input_left;
      plan.transpose_input_right  = list_configs[selected_config].trans_input_right;

      plan.info_trans_A_perm      = list_configs[selected_config].info_perms_input_left;
      plan.info_trans_B_perm      = list_configs[selected_config].info_perms_input_right;
      
      plan.gemm_trans_A           = list_configs[selected_config].gemm_trans_A;
      plan.gemm_trans_B           = list_configs[selected_config].gemm_trans_B;
    }

    plan.transpose_output       = list_configs[selected_config].trans_output;
    plan.info_trans_C_perm      = list_configs[selected_config].info_perms_output;
    plan.info_trans_C_dim       = list_configs[selected_config].info_dims_output;
  } 
  else 
  {
    plan.swap_AB                = list_configs[0].swap;
    if (plan.swap_AB == true)
    {
      plan.gemm_m = len_gemm_n;
      plan.gemm_n = len_gemm_m;

      fill_dims_from_modes(plan.info_trans_A_dim, modeB, extent);
      fill_dims_from_modes(plan.info_trans_B_dim, modeA, extent);

      plan.transpose_input_left   = list_configs[0].trans_input_right;
      plan.transpose_input_right  = list_configs[0].trans_input_left;

      plan.info_trans_A_perm      = list_configs[0].info_perms_input_right;
      plan.info_trans_B_perm      = list_configs[0].info_perms_input_left;

      plan.gemm_trans_A           = list_configs[0].gemm_trans_B;
      plan.gemm_trans_B           = list_configs[0].gemm_trans_A;
    }
    // (plan.swap_AB == false)
    else 
    {
      plan.gemm_m = len_gemm_m;
      plan.gemm_n = len_gemm_n;

      fill_dims_from_modes(plan.info_trans_A_dim, modeA, extent);
      fill_dims_from_modes(plan.info_trans_B_dim, modeB, extent);

      plan.transpose_input_left   = list_configs[0].trans_input_left;
      plan.transpose_input_right  = list_configs[0].trans_input_right;

      plan.info_trans_A_perm      = list_configs[0].info_perms_input_left;
      plan.info_trans_B_perm      = list_configs[0].info_perms_input_right;

      plan.gemm_trans_A           = list_configs[0].gemm_trans_A;
      plan.gemm_trans_B           = list_configs[0].gemm_trans_B;
    }

    plan.transpose_output       = list_configs[0].trans_output;    
    plan.info_trans_C_perm      = list_configs[0].info_perms_output;
    plan.info_trans_C_dim       = list_configs[0].info_dims_output;
  }
}


// #define OPT_TIME
// #define OPT_CUTT_PLAN_MEASURE // 

/**
 * @brief TTGT execution based on the planned configurations
 * 
 */
template <typename T>
void ttgt_cuTT_prepare(const ttgt_cuTT_handle& plan,
  ttgt_cuTT_runtime& runtime,
  T* d_A, T* d_A_trans, T* d_B, T* d_B_trans, T* d_C, T* d_C_trans)
{
  check_cublas_or_throw(cublasCreate(&runtime.cublas_handle), "cublasCreate");
	cublasSetMathMode(runtime.cublas_handle, CUBLAS_TENSOR_OP_MATH);

  double tmp_dgemm_alpha          = 1.0;
  double tmp_dgemm_beta           = 0.0;

  T* dummy_dev_tensor_A = nullptr;
  T* dummy_dev_tensor_B = nullptr;
  T* dummy_dev_tensor_C = nullptr;

  check_cuda_or_throw(cudaMalloc((void**)&dummy_dev_tensor_C, sizeof(T) * 10000), "cudaMalloc(dummy C)");
  check_cuda_or_throw(cudaMalloc((void**)&dummy_dev_tensor_A, sizeof(T) * 10000), "cudaMalloc(dummy A)");
  check_cuda_or_throw(cudaMalloc((void**)&dummy_dev_tensor_B, sizeof(T) * 10000), "cudaMalloc(dummy B)");

  for (int i = 0; i < 10; i++) {
    check_cublas_or_throw(cublasGemmEx(runtime.cublas_handle,
      CUBLAS_OP_N, CUBLAS_OP_N, 
      100, 100, 100, 
      &tmp_dgemm_alpha, 
      dummy_dev_tensor_A, CUDA_R_64F, 100,
      dummy_dev_tensor_B, CUDA_R_64F, 100,
      &tmp_dgemm_beta, 
      dummy_dev_tensor_C, CUDA_R_64F, 100, 
      CUBLAS_COMPUTE_64F,       // Forces FP64 accumulate
      CUBLAS_GEMM_DEFAULT_TENSOR_OP // Explicit Tensor Core request
    ), "cublasGemmEx(dummy)");
  }
  check_cuda_or_throw(cudaFree(dummy_dev_tensor_A), "cudaFree(dummy A)");
  check_cuda_or_throw(cudaFree(dummy_dev_tensor_B), "cudaFree(dummy B)");
  check_cuda_or_throw(cudaFree(dummy_dev_tensor_C), "cudaFree(dummy C)");

  T* exec_A = d_A;
  T* exec_A_trans = d_A_trans;
  T* exec_B = d_B;
  T* exec_B_trans = d_B_trans;
  if (plan.swap_AB == true) {
    std::swap(exec_A, exec_B);
    std::swap(exec_A_trans, exec_B_trans);
  }

  runtime.exec_A = exec_A;
  runtime.exec_A_trans = exec_A_trans;
  runtime.exec_B = exec_B;
  runtime.exec_B_trans = exec_B_trans;

  check_cuda_or_throw(cudaEventCreate(&runtime.start_trans_A), "cudaEventCreate(start_trans_A)");
  check_cuda_or_throw(cudaEventCreate(&runtime.stop_trans_A), "cudaEventCreate(stop_trans_A)");
  check_cuda_or_throw(cudaEventCreate(&runtime.start_trans_B), "cudaEventCreate(start_trans_B)");
  check_cuda_or_throw(cudaEventCreate(&runtime.stop_trans_B), "cudaEventCreate(stop_trans_B)");
  check_cuda_or_throw(cudaEventCreate(&runtime.start_trans_C), "cudaEventCreate(start_trans_C)");
  check_cuda_or_throw(cudaEventCreate(&runtime.stop_trans_C), "cudaEventCreate(stop_trans_C)");
  check_cuda_or_throw(cudaEventCreate(&runtime.start_gemm), "cudaEventCreate(start_gemm)");
  check_cuda_or_throw(cudaEventCreate(&runtime.stop_gemm), "cudaEventCreate(stop_gemm)");
  runtime.has_events = true;

  cuttCheck(cuttPlanMeasure(&runtime.cutt_handle_A, plan.info_trans_A_dim.size(), const_cast<int*>(plan.info_trans_A_dim.data()), const_cast<int*>(plan.info_trans_A_perm.data()), sizeof(T), 0, exec_A, exec_A_trans));
  runtime.has_cutt_handle_A = true;
  cuttCheck(cuttPlanMeasure(&runtime.cutt_handle_B, plan.info_trans_B_dim.size(), const_cast<int*>(plan.info_trans_B_dim.data()), const_cast<int*>(plan.info_trans_B_perm.data()), sizeof(T), 0, exec_B, exec_B_trans));
  runtime.has_cutt_handle_B = true;
  cuttCheck(cuttPlanMeasure(&runtime.cutt_handle_C, plan.info_trans_C_dim.size(), const_cast<int*>(plan.info_trans_C_dim.data()), const_cast<int*>(plan.info_trans_C_perm.data()), sizeof(T), 0, d_C, d_C_trans));
  runtime.has_cutt_handle_C = true;
}

template <typename T>
void ttgt_cuTT_execute(const ttgt_cuTT_handle& plan,
  ttgt_cuTT_runtime& runtime,
  T*& d_C, T* d_C_trans)
{
  runtime.last_transpose_a_ms = 0.0f;
  runtime.last_transpose_b_ms = 0.0f;
  runtime.last_gemm_ms = 0.0f;
  runtime.last_transpose_c_ms = 0.0f;
  runtime.last_total_ms = 0.0f;

  T* d_A_trans = static_cast<T*>(runtime.exec_A_trans);
  T* d_B_trans = static_cast<T*>(runtime.exec_B_trans);
  T* exec_A = static_cast<T*>(runtime.exec_A);
  T* exec_B = static_cast<T*>(runtime.exec_B);

  const double gemm_alpha = 1.0;
  const double gemm_beta = 0.0;

#define ENABLE_CUBLAS_GEMMEX

  if (plan.transpose_input_left == true)
  {
    check_cuda_or_throw(cudaEventRecord(runtime.start_trans_A), "cudaEventRecord(start_trans_A)");
    cuttCheck(cuttExecute(runtime.cutt_handle_A, exec_A, d_A_trans));
    check_cuda_or_throw(cudaEventRecord(runtime.stop_trans_A), "cudaEventRecord(stop_trans_A)");
    check_cuda_or_throw(cudaEventSynchronize(runtime.stop_trans_A), "cudaEventSynchronize(stop_trans_A)");
    check_cuda_or_throw(cudaEventElapsedTime(&runtime.last_transpose_a_ms, runtime.start_trans_A, runtime.stop_trans_A), "cudaEventElapsedTime(transpose A)");
  }
  else { d_A_trans = exec_A; }

  if (plan.transpose_input_right == true)
  {
    check_cuda_or_throw(cudaEventRecord(runtime.start_trans_B), "cudaEventRecord(start_trans_B)");
    cuttCheck(cuttExecute(runtime.cutt_handle_B, exec_B, d_B_trans));
    check_cuda_or_throw(cudaEventRecord(runtime.stop_trans_B), "cudaEventRecord(stop_trans_B)");
    check_cuda_or_throw(cudaEventSynchronize(runtime.stop_trans_B), "cudaEventSynchronize(stop_trans_B)");
    check_cuda_or_throw(cudaEventElapsedTime(&runtime.last_transpose_b_ms, runtime.start_trans_B, runtime.stop_trans_B), "cudaEventElapsedTime(transpose B)");
  }
  else { d_B_trans = exec_B; }

  check_cuda_or_throw(cudaEventRecord(runtime.start_gemm), "cudaEventRecord(start_gemm)");
  if (plan.gemm_trans_A == false)
  {
    if (plan.gemm_trans_B == false)
    {
    #ifdef ENABLE_CUBLAS_GEMMEX
      check_cublas_or_throw(cublasGemmEx(runtime.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                plan.gemm_m, plan.gemm_n, plan.gemm_k,
                                &gemm_alpha,  d_A_trans, CUDA_R_64F, plan.gemm_m,
                                              d_B_trans, CUDA_R_64F, plan.gemm_k,
                                              &gemm_beta,
                                              d_C_trans, CUDA_R_64F, plan.gemm_m,
                                              CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "cublasGemmEx(N,N)");
    #endif
    }
    else 
    {
    #ifdef ENABLE_CUBLAS_GEMMEX
      check_cublas_or_throw(cublasGemmEx(runtime.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                plan.gemm_m, plan.gemm_n, plan.gemm_k,
                                &gemm_alpha,  d_A_trans, CUDA_R_64F, plan.gemm_m,
                                              d_B_trans, CUDA_R_64F, plan.gemm_n,
                                              &gemm_beta,
                                              d_C_trans, CUDA_R_64F, plan.gemm_m,
                                              CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "cublasGemmEx(N,T)");
    #endif
    }
  }
  else 
  {
    if (plan.gemm_trans_B == false)
    {
    #ifdef ENABLE_CUBLAS_GEMMEX
      check_cublas_or_throw(cublasGemmEx(runtime.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                plan.gemm_m, plan.gemm_n, plan.gemm_k,
                                &gemm_alpha,  d_A_trans, CUDA_R_64F, plan.gemm_k,
                                              d_B_trans, CUDA_R_64F, plan.gemm_k,
                                              &gemm_beta,
                                              d_C_trans, CUDA_R_64F, plan.gemm_m,
                                              CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "cublasGemmEx(T,N)");
    #endif 
    }
    else 
    {
    #ifdef ENABLE_CUBLAS_GEMMEX
      check_cublas_or_throw(cublasGemmEx(runtime.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T,
                                plan.gemm_m, plan.gemm_n, plan.gemm_k,
                                &gemm_alpha,  d_A_trans, CUDA_R_64F, plan.gemm_k,
                                              d_B_trans, CUDA_R_64F, plan.gemm_n,
                                              &gemm_beta,
                                              d_C_trans, CUDA_R_64F, plan.gemm_m,
                                              CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "cublasGemmEx(T,T)");
    #endif
    }
  }
  check_cuda_or_throw(cudaEventRecord(runtime.stop_gemm), "cudaEventRecord(stop_gemm)");
  check_cuda_or_throw(cudaEventSynchronize(runtime.stop_gemm), "cudaEventSynchronize(stop_gemm)");
  check_cuda_or_throw(cudaEventElapsedTime(&runtime.last_gemm_ms, runtime.start_gemm, runtime.stop_gemm), "cudaEventElapsedTime(gemm)");

  if (plan.transpose_output == true)
  {
    check_cuda_or_throw(cudaEventRecord(runtime.start_trans_C), "cudaEventRecord(start_trans_C)");
    cuttExecute(runtime.cutt_handle_C, d_C_trans, d_C);
    check_cuda_or_throw(cudaEventRecord(runtime.stop_trans_C), "cudaEventRecord(stop_trans_C)");
    check_cuda_or_throw(cudaEventSynchronize(runtime.stop_trans_C), "cudaEventSynchronize(stop_trans_C)");
    check_cuda_or_throw(cudaEventElapsedTime(&runtime.last_transpose_c_ms, runtime.start_trans_C, runtime.stop_trans_C), "cudaEventElapsedTime(transpose C)");
  }
  else { d_C = d_C_trans; }

  runtime.last_total_ms =
    runtime.last_transpose_a_ms +
    runtime.last_transpose_b_ms +
    runtime.last_gemm_ms +
    runtime.last_transpose_c_ms;
}

/**
 * @brief 
 * 
 * @param plan 
 */
void ttgt_cuTT_plan_destroy(ttgt_cuTT_handle& plan)
{
  // delete[] plan.list_configs;
  // plan.list_configs = nullptr;
}

void ttgt_cuTT_runtime_destroy(ttgt_cuTT_runtime& runtime)
{
  if (runtime.has_events) {
    cudaEventDestroy(runtime.start_trans_A);
    cudaEventDestroy(runtime.stop_trans_A);
    cudaEventDestroy(runtime.start_trans_B);
    cudaEventDestroy(runtime.stop_trans_B);
    cudaEventDestroy(runtime.start_trans_C);
    cudaEventDestroy(runtime.stop_trans_C);
    cudaEventDestroy(runtime.start_gemm);
    cudaEventDestroy(runtime.stop_gemm);
    runtime.start_trans_A = nullptr;
    runtime.stop_trans_A = nullptr;
    runtime.start_trans_B = nullptr;
    runtime.stop_trans_B = nullptr;
    runtime.start_trans_C = nullptr;
    runtime.stop_trans_C = nullptr;
    runtime.start_gemm = nullptr;
    runtime.stop_gemm = nullptr;
    runtime.has_events = false;
  }
  if (runtime.has_cutt_handle_A) {
    cuttCheck(cuttDestroy(runtime.cutt_handle_A));
    runtime.has_cutt_handle_A = false;
  }
  if (runtime.has_cutt_handle_B) {
    cuttCheck(cuttDestroy(runtime.cutt_handle_B));
    runtime.has_cutt_handle_B = false;
  }
  if (runtime.has_cutt_handle_C) {
    cuttCheck(cuttDestroy(runtime.cutt_handle_C));
    runtime.has_cutt_handle_C = false;
  }
  if (runtime.cublas_handle != nullptr) {
    cublasDestroy(runtime.cublas_handle);
    runtime.cublas_handle = nullptr;
  }
  runtime.exec_A = nullptr;
  runtime.exec_A_trans = nullptr;
  runtime.exec_B = nullptr;
  runtime.exec_B_trans = nullptr;
}

// Explicit template instantiation for double data type
template void ttgt_cuTT_prepare<double>(const ttgt_cuTT_handle& plan, ttgt_cuTT_runtime& runtime, double* d_A, double* d_A_trans, double* d_B, double* d_B_trans, double* d_C, double* d_C_trans);
template void ttgt_cuTT_execute<double>(const ttgt_cuTT_handle& plan, ttgt_cuTT_runtime& runtime, double*& d_C, double* d_C_trans);
