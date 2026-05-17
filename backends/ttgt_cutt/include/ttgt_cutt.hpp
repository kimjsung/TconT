#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <locale.h>
#include <dlfcn.h>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cutt.h>
#include <time.h>

#include <vector>
#include <unordered_map>

#define A100_40GB_MiB   38147 // 40GB in MiB


#define HW_SCRATCH          2000             // MiB 
#define HW_PERCENTAGE       20              // % of |A|, |B| and |C|

#define SIZE_FLOAT          4
#define SIZE_DOUBLE         8
#define SIZE_GLOBAL_MEMORY  A100_40GB_MiB
#define SIZE_MB             1048576         // MiB
#define SIZE_TYPE           SIZE_DOUBLE
#define SIZE_NAME           4

#define LENGTH_TMP_INDICES  10
#define SIZE_IDX_DEFAULT    16

#define OPT_TRANSPOSE_YES   1
#define OPT_TRANSPOSE_NO   -1

#define TYPE_TENSOR_A       1
#define TYPE_TENSOR_B       2
#define TYPE_TENSOR_C       0

#define TYPE_AB             0
#define TYPE_BA             1

#define SCRATCH_FIT         1
#define SCRATCH_UNFIT      -1

#define GIVEN_TENSORS_YES   1
#define GIVEN_TENSORS_NO   -1

#define TYPE_EXTERNAL       1
#define TYPE_INTERNAL       0

#define TYPE_DIFF_ENUM      1
#define TYPE_SAME_ENUM     -1



//
typedef struct info_index
{
    char    name[SIZE_NAME]; 
    int     size;
    int     tile_size;
    int     idx_type;
}idx;

//
typedef struct info_tensor_contraction
{
    int     len_output;
    int     len_input_left;
    int     len_input_right;
    int     len_internal_indices;

    idx*    info_output;
    idx*    info_input_left;
    idx*    info_input_right;
    idx*    info_internal_indices;

    int     op;             // 1: +=, 2: -=

    int     size_output;
    int     size_input_left;
    int     size_input_right;
}tc;

//
typedef struct info_tiles
{
    idx*    info_slice;

    int     size_output_slice;
    int     size_input_left_slice;
    int     size_input_right_slice;

    int     dgemm_m;
    int     dgemm_n;
    int     dgemm_k;

    info_tiles*     next_info_tiles;
    info_tiles*     prev_info_tiles;
}tiles;

//
typedef struct info_transpose
{
    int     transpose;
    int     dgemm_transpose;

    int     len_indice;
    int*    ttlg_dims;
    int*    ttlg_perms;
}tt;

// 
typedef struct info_transs
{
    int     transpose;
    int     dgemm_transpose;

    int     len_indice;
    int*    ttlg_dims;
    int*    ttlg_perms;
}info_trans;


/**
 * @brief NEW configuration structure for TTGT
 * 
 */
typedef struct new_info_config
{
    bool swap; // false: AB, true: BA
    bool useless = true; // 

    bool trans_input_left;  // T: A -> A'
    bool trans_input_right; // T: B -> B'
    bool trans_output;      // T: C' -> C

    std::vector<int> info_perms_input_left;
    std::vector<int> info_perms_input_right;

    std::vector<int> info_perms_output; // C'
    std::vector<int> info_dims_output;  // C'

    // false: N, true; T
    bool gemm_trans_A = false;
    bool gemm_trans_B = false;

} new_config;

//
typedef struct info_configuration
{
    int     total_cost;
    
    //  for lottgt
    int     cost_cudaMemcpy_input_left;
    int     cost_cudaMemcpy_input_right;
    int     cost_cudaMemcpy_output;

    //  for ttgt
    int     cost_transpose_input_left;
    int     cost_transpose_input_right;
    int     cost_dgemm;
    int     cost_transpose_output;

    //  transpositions for A, B and C
    int     info_swap;

    int     transpose_input_left;
    int     transpose_input_right;
    int     transpose_output;

    //
    int*    info_perms_input_left;
    int*    info_perms_input_right;
    int*    info_perms_output;

    //
    int*    info_dims_output;
}info_config;


// temporary
struct ttgt_cuTT_handle {

    // info. for trans
    bool swap_AB = false; // whether to swap A and B
    bool transpose_input_left = false; // whether to transpose input left tensor
    bool transpose_input_right = false; // whether to transpose input right tensor
    bool transpose_output = false; // whether to transpose output tensor

    std::vector<int> info_trans_A_dim;  // A
    std::vector<int> info_trans_A_perm; // A'

    std::vector<int> info_trans_B_dim;  // B
    std::vector<int> info_trans_B_perm; // B'

    std::vector<int> info_trans_C_dim;  // C'
    std::vector<int> info_trans_C_perm; // C

    // info. for gemm
    int gemm_m = 1;
    int gemm_n = 1;
    int gemm_k = 1;

    // false: N, true; T
    bool gemm_trans_A = false; 
    bool gemm_trans_B = false; 

    // 
    // int num_configs; //
    // new_config* list_configs; // 
};

struct ttgt_cuTT_runtime {
    cublasHandle_t cublas_handle = nullptr;
    cuttHandle cutt_handle_A = 0;
    cuttHandle cutt_handle_B = 0;
    cuttHandle cutt_handle_C = 0;
    bool has_cutt_handle_A = false;
    bool has_cutt_handle_B = false;
    bool has_cutt_handle_C = false;
};


/**
 * @brief Main APIs for TTGT-cuTT
 * 
 */
void ttgt_cuTT_plan(ttgt_cuTT_handle& plan, 
    const std::vector<char>& modeC, 
    const std::vector<char>& modeA, 
    const std::vector<char>& modeB, 
    const std::unordered_map<char, int64_t>& extent, 
    unsigned int target_config = 0);

template <typename T>
void ttgt_cuTT_prepare(
    const ttgt_cuTT_handle& plan,
    ttgt_cuTT_runtime& runtime,
    T* d_A, T* d_A_trans,
    T* d_B, T* d_B_trans,
    T* d_C, T* d_C_trans);

template <typename T>
void ttgt_cuTT_execute(const ttgt_cuTT_handle& plan, 
    ttgt_cuTT_runtime& runtime,
    T* d_A, T* d_A_trans, 
    T* d_B, T* d_B_trans, 
    T*& d_C, T* d_C_trans);

void ttgt_cuTT_plan_destroy(ttgt_cuTT_handle& plan);
void ttgt_cuTT_runtime_destroy(ttgt_cuTT_runtime& runtime);

//  helper
tc*     create_tc(char* string_tensor_contraction, int* int_problem_size, unsigned long long int* num_operations);
void    print_tensor(const char* tensor_name, idx* info_tensor, int size);
void    delete_example(tc* target);
void    swap(int* array, int i, int j);
void    perm(int* array, int n, int i, int* offset, int* output);
int     find_size(const char* name, idx* target_list, int target_list_size);
int     find_index(const char* name, idx* target_list, int target_list_size);
int     factorial(int target);

//
//  LoTTGT
//
void    tiling_scratch(tc* info_tc);
void    lottgt_enumeration(tc* info_tc, idx* info_tiling, int* tiles, int idx_bound);
void    helper_recursive_tiling_idx(tc* info_tc, int idx_offset, int idx_bound, idx* info_tiling, int* tiles);
int     helper_tiling_fit_scratch(tc* info_tc, idx* info_tiling, int* tiles, int idx_bound);
int     helper_ttgt_vs_lottgt(tc* info_tc, tt* picked_config_output, tt* picked_config_input_left, tt* picked_config_input_right, int opt_keep_tensors);

//  TTGT
int     ttgt_enumeration(tc* info_tc, tiles*& result_tiles, tt*& result_tt_output, tt*& result_tt_input_left, tt*& result_tt_input_right, int target_configuration, int opt_manual);

//  model-base
int     tctt_model_overall(tc* info_tc, tiles*& result_tiles, tt*& result_tt_output, tt*& result_tt_input_left, tt*& result_tt_input_right, int target_configuration);
int     tctt_model_extract_tensor();//idx* target_tensor, tiles* info_tiles);
int     tctt_model_transpose_tensor();
int     tctt_model_dgemm(int row_A, int column_A, int trans_A, int row_B, int column_B, int trans_B);

//
void    check_simple_four_cases(int num_perms_trans, info_config* list_configurations, tc* info_tc);


//
int     model_base(int num_configurations, info_config* list_configurations, tc* info_tc);
int     model_ttgt_ttlg(info_config a_config, int opt);
int     model_ttgt_gemm(info_config a_config);

//
void    lottgt_tensor_copy(double* tensor_input, double* tensor_output);
void    lottgt_tensor_store(double* tensor_input, double* tensor_output);

//
int     check_Constraints(int size_c_slice, int is_t_c, int size_a_slice, int is_t_a, int size_b_slice, int is_t_b, float mem_size_scratch);
