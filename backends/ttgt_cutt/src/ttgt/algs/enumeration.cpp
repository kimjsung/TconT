#include "../ttgt_ttlg.h"

#define SIZE_SWAP_INPUTS    2

// #define DEBUG_PERMUTATIONS
// #define DEBUG_PERMUTATIONS_DETAIL
// #define DEBUG_PERMUTATIONS_DETAIL_INNER
#define DEBUG_PERMUTATIONS_RESULT
// #define DEBUG_PROBLEM_SIZE
// #define DEBUG_MAKE_INFO_TTLG
// #define DEBUG_PERMUTATIONS_ENUMERATE

// 
// 
// 
int ttlg_new_enumeration(tc* info_tc)
{
    int  len_output         = info_tc->len_output;
    int  len_input_left     = info_tc->len_input_left;
    int  len_input_right    = info_tc->len_input_right;
    idx* info_output        = info_tc->info_output;
    idx* info_input_left    = info_tc->info_input_left;
    idx* info_input_right   = info_tc->info_input_right;

    return 0;
}

//
//  inputs: info_tc, target_configuration
//  outputs: result_tiles, result_tt_output, result_tt_input_left, result_tt_input_right
//
int ttgt_enumeration(tc* info_tc, tiles*& result_tiles, tt*& result_tt_output, tt*& result_tt_input_left, tt*& result_tt_input_right, int target_configuration, int opt_manual)
{
    //
    //  Given Information of Tensor Contraction.
    //
    int  len_output         = info_tc->len_output;
    int  len_input_left     = info_tc->len_input_left;
    int  len_input_right    = info_tc->len_input_right;
    idx* info_output        = info_tc->info_output;
    idx* info_input_left    = info_tc->info_input_left;
    idx* info_input_right   = info_tc->info_input_right;

    // Need to Create Tile-Sizes which follows the constraints from the Full.
    int len_external_indices = info_tc->len_output;
    int len_internal_indices = ((info_tc->len_input_left + info_tc->len_input_right) - len_external_indices) / 2;

    idx* problem_size_external = (idx*)malloc(sizeof(idx) * len_external_indices);
    idx* problem_size_internal = (idx*)malloc(sizeof(idx) * len_internal_indices);

    for (int i = 0; i < len_external_indices; i++)
    {
        strncpy(problem_size_external[i].name, info_tc->info_output[i].name, SIZE_NAME);
        problem_size_external[i].size = info_tc->info_output[i].size;
        // printf ("%d ", info_tc->info_output[i].size);
    }
    // printf ("\n");

    for (int i = 0, int_i = 0; i < info_tc->len_input_left; i++)
    {
        if (find_index(info_tc->info_input_left[i].name, info_tc->info_output, info_tc->len_output) == -1)
        {
            strncpy(problem_size_internal[int_i].name, info_tc->info_input_left[i].name, SIZE_NAME);
            problem_size_internal[int_i++].size = info_tc->info_input_left[i].size;
        }
    }

#ifdef DEBUG_PROBLEM_SIZE
    printf ("[%s] Problem_Size\n", __func__);
    printf ("[%s] External: ", __func__);
    for (int i = 0; i < len_external_indices; i++)
        printf ("%s (%d), ", problem_size_external[i].name, problem_size_external[i].size);
    printf ("\n");
    printf ("[%s] Internal: ", __func__);
    for (int i = 0; i < len_internal_indices; i++)
        printf ("%s (%d), ", problem_size_internal[i].name, problem_size_internal[i].size);
    printf ("\n");
#endif

    //
    int size_output         = 1;
    int size_input_left     = 1;
    int size_input_right    = 1;
    int size_dgemm_m        = 1;
    int size_dgemm_n        = 1;
    int size_dgemm_k        = 1;

    //
    //  related to LoTTGT
    //
    //  [Slice] Output
    for (int i = 0; i < len_output; i++)
    {
        if (find_size(info_output[i].name, problem_size_external, len_external_indices) != 0)
        {
            size_output *= find_size(info_output[i].name, problem_size_external, len_external_indices);
        }
    }
    
    //  [Slice] Input-Left
    for (int i = 0; i < len_input_left; i++)
    {
        if (find_size(info_input_left[i].name, problem_size_external, len_external_indices) != 0)
        {
            int temp = find_size(info_input_left[i].name, problem_size_external, len_external_indices);
            size_input_left   *= temp;
            size_dgemm_m      *= temp;
        }

        if (find_size(info_input_left[i].name, problem_size_internal, len_internal_indices) != 0)
        {
            int temp = find_size(info_input_left[i].name, problem_size_internal, len_internal_indices);
            size_input_left   *= temp;
            size_dgemm_k      *= temp;
        }
    }

    //  [Slice] Input-Right
    for (int i = 0; i < len_input_right; i++)
    {
        if (find_size(info_input_right[i].name, problem_size_external, len_external_indices) != 0)
        {
            int temp = find_size(info_input_right[i].name, problem_size_external, len_external_indices);
            size_input_right  *= temp;
            size_dgemm_n      *= temp;
        }

        if (find_size(info_input_right[i].name, problem_size_internal, len_internal_indices) != 0)
        {
            size_input_right  *= find_size(info_input_right[i].name, problem_size_internal, len_internal_indices);
        }
    }
    
    //
    //  result_tiles (This Process Might be Combined with the Below Process.)
    //
    result_tiles                            = (tiles*)malloc(sizeof(tiles));
    result_tiles->info_slice                = (idx*)malloc(sizeof(idx) * (len_external_indices + len_internal_indices));
    result_tiles->size_output_slice         = size_output;
    result_tiles->size_input_left_slice     = size_input_left;
    result_tiles->size_input_right_slice    = size_input_right;
    result_tiles->dgemm_m                   = size_dgemm_m;
    result_tiles->dgemm_n                   = size_dgemm_n;
    result_tiles->dgemm_k                   = size_dgemm_k;

    // printf ("[%s] Problem_Size for GEMM: m: %d, n: %d, k: %d\n", __func__, result_tiles->dgemm_m, result_tiles->dgemm_n, result_tiles->dgemm_k);
    // printf ("===========================================================================\n");

    //
    //  Permutations for Tensor Transpositions.
    //
    for (int i = 0; i < len_external_indices; i++)
    {
        strncpy(result_tiles->info_slice[i].name, problem_size_external[i].name, SIZE_NAME);
        result_tiles->info_slice[i].size = problem_size_external[i].size;
    }
    for (int i = 0; i < len_internal_indices; i++)
    {
        strncpy(result_tiles->info_slice[i + len_external_indices].name, problem_size_internal[i].name, SIZE_NAME);
        result_tiles->info_slice[i + len_external_indices].size = problem_size_internal[i].size;
    }

    //
    //  [1-1] Input-Left
    //
    int num_ext_input_left = 0;
    int num_int_input_left = 0;
    for (int i = 0; i < len_input_left; i++)
    {
        if (find_size(info_input_left[i].name, info_output, len_output) != 0)
            num_ext_input_left++;
        else
            num_int_input_left++;
    }
#ifdef DEBUG_PERMUTATIONS
    printf ("[Input][Left]  # of External Index: %d, # of Internal Index: %d\n", num_ext_input_left, num_int_input_left);
#endif

    //
    int idx_ext_input_left[num_ext_input_left];
    int idx_int_input_left[num_int_input_left];
    for (int i = 0, idx_ext = 0, idx_int = 0; i < len_input_left; i++)
    {
        if (find_size(info_input_left[i].name, info_output, len_output) != 0)
            idx_ext_input_left[idx_ext++] = i;
        else
            idx_int_input_left[idx_int++] = i;
    }

#ifdef DEBUG_PERMUTATIONS
    for (int i = 0; i < num_ext_input_left; i++)
        printf ("[LEFT][EXT][%d] %d\n", i, idx_ext_input_left[i]);
    
    for (int i = 0; i < num_int_input_left; i++)
        printf ("[LEFT][INT][%d] %d\n", i, idx_int_input_left[i]);
#endif

    int     num_perms_ext_input_left = factorial(num_ext_input_left);
    int     num_perms_int_input_left = factorial(num_int_input_left);
    int*    idx_perms_ext_input_left = (int*)malloc(sizeof(int) * num_perms_ext_input_left * num_ext_input_left);
    int*    idx_perms_int_input_left = (int*)malloc(sizeof(int) * num_perms_int_input_left * num_int_input_left);    // Common for Both inputs

    //
    //  [1-1-1] Create Perms. of External Indices in Input-Left
    //
    int offset = 0;
    perm(idx_ext_input_left, num_ext_input_left, 0, &offset, idx_perms_ext_input_left);
#ifdef DEBUG_PERMUTATIONS
    for (int i = 0; i < num_perms_ext_input_left; i++)
    {
        printf ("[LEFT][PERMUTATION][EXT][%d] ", i);
        for (int j = 0; j < num_ext_input_left; j++)
        {
            printf ("%d, ", idx_perms_ext_input_left[i * num_ext_input_left + j]);
        }
        printf ("\n");
    }
#endif

    //
    //  [1-1-2] Create Perms. of Internal Indices in Input-Left (will be used in Input-Right)
    //
    offset = 0;
    perm(idx_int_input_left, num_int_input_left, 0, &offset, idx_perms_int_input_left);
#ifdef DEBUG_PERMUTATIONS
    for (int i = 0; i < num_perms_int_input_left; i++)
    {
        printf ("[LEFT][PERMUTATION][INT][%d] ", i);
        for (int j = 0; j < num_int_input_left; j++)
        {
            printf ("%d, ", idx_perms_int_input_left[i * num_int_input_left + j]);
        }
        printf ("\n");
    }
#endif

    //
    //  [1-2] Input-Right
    //
    int num_ext_input_right = 0;
    int num_int_input_right = 0;
    for (int i = 0; i < len_input_right; i++)
    {
        if (find_size(info_input_right[i].name, info_output, len_output) != 0)
            num_ext_input_right++;
        else
            num_int_input_right++;
    }
#ifdef DEBUG_PERMUTATIONS
    printf ("[Input][Right] # of External Index: %d, # of Internal Index: %d\n", num_ext_input_right, num_int_input_right);
#endif

    //
    int idx_ext_input_right[num_ext_input_right];
    int idx_int_input_right[num_int_input_right];
    for (int i = 0, idx_ext = 0, idx_int = 0; i < len_input_right; i++)
    {
        if (find_size(info_input_right[i].name, info_output, len_output) != 0)
            idx_ext_input_right[idx_ext++] = i;
        else
            idx_int_input_right[idx_int++] = i;
    }
#ifdef DEBUG_PERMUTATIONS
    for (int i = 0; i < num_ext_input_right; i++)
        printf ("[RIGHT][EXT][%d] %d\n", i, idx_ext_input_right[i]);
    
    for (int i = 0; i < num_int_input_right; i++)
        printf ("[RIGHT][INT][%d] %d\n", i, idx_int_input_right[i]);
#endif

    int     num_perms_ext_input_right = factorial(num_ext_input_right);
    int     num_perms_int_input_right = factorial(num_int_input_right);
    int*    idx_perms_ext_input_right = (int*)malloc(sizeof(int) * num_perms_ext_input_right * num_ext_input_right); // 2D Array: # of Permutation * # of Indice
    int*    idx_perms_int_input_right = (int*)malloc(sizeof(int) * num_perms_int_input_right * num_int_input_right); // 2D Array: # of Permutation * # of Indice

    //
    //  [1-2-1] Create Perms. of External Indices in Input-Right
    //
    offset = 0;
    perm(idx_ext_input_right, num_ext_input_right, 0, &offset, idx_perms_ext_input_right);
#ifdef DEBUG_PERMUTATIONS
    for (int i = 0; i < num_perms_ext_input_right; i++)
    {
        printf ("[RIHGT][PERMUTATION][EXT][%d] ", i);
        for (int j = 0; j < num_ext_input_right; j++)
        {
            printf ("%d, ", idx_perms_ext_input_right[i * num_ext_input_right + j]);
        }
        printf ("\n");
    }
#endif

    //
    //  RIGHT-Tensor's INT == LEFT-Tensor's INT
    //
    for (int i = 0; i < num_perms_int_input_left; i++)
    {
        for (int j = 0; j < num_int_input_left; j++)
        {
            for (int k = 0; k < num_ext_input_right + num_int_input_right; k++)
            {
                if (strcmp(info_input_right[k].name, info_input_left[idx_perms_int_input_left[i * num_int_input_left + j]].name) == 0)
                {
                    idx_perms_int_input_right[i * num_int_input_right + j] = k;
                }
            }
        }
    }

#ifdef DEBUG_PERMUTATIONS
    for (int i = 0; i < num_perms_int_input_right; i++)
    {
        printf ("[RIHGT][PERMUTATION][INT][%d] ", i);
        for (int j = 0; j < num_int_input_right; j++)
        {
            printf ("%d, ", idx_perms_int_input_right[i * num_int_input_right + j]);
        }
        printf ("\n");
    }
#endif

    //  [Exception]   
    if (num_int_input_right != num_int_input_left)
    {
        printf ("[%s] ERROR: Input Tensors have different number of internal indices\n", __func__);
    }

    //  [Exception]
    if ((num_ext_input_left + num_ext_input_right) != len_output)
    {
        printf ("[%s] ERROR: the number of external indices in both inputs is different from the output's ones\n", __func__);
    }

    //
    //  # of Permutations
    //
    int num_perms_trans         = factorial(num_ext_input_left) * 2 * 
                                  factorial(num_int_input_left) * 
                                  factorial(num_ext_input_right) * 2;
    int num_perms_trans_swapped = 2 * num_perms_trans;
#ifdef DEBUG_PERMUTATIONS
    printf ("[Calculated] There are %d different ways to transpose inputs.\n", num_perms_trans);
    printf ("[Calculated] %d (AB | BA)\n", num_perms_trans_swapped);
#endif

    //
    //  [1-3] Create a Target Output
    //
    int target_output[len_output][2];           // 0: Left, 1: Right
    int target_output_swapped[len_output][2];   // 1: Left, 0: Right
    int temp;
    for (int i = 0; i < len_output; i++)
    {
        temp = find_index(info_output[i].name, info_input_left, len_input_left);
        if (temp != -1)
        {
            target_output[i][0] = 0;                // Input-Left
            target_output[i][1] = temp;

            target_output_swapped[i][0] = 1;
            target_output_swapped[i][1] = temp;
            // printf ("[AB][%d] %d, %d\n", i, 0, temp);
            // printf ("[BA][%d] %d, %d\n", i, 1, temp);
        }

        temp = find_index(info_output[i].name, info_input_right, len_input_right);
        if (temp != -1)
        {
            target_output[i][0] = 1;                // Input-Right
            target_output[i][1] = temp;

            target_output_swapped[i][0] = 0;        //
            target_output_swapped[i][1] = temp;
            // printf ("[AB][%d] %d, %d\n", i, 1, temp);
            // printf ("[BA][%d] %d, %d\n", i, 0, temp);
        }
    }

#ifdef DEBUG_PERMUTATIONS
    printf ("==========================================================================================\n");
    printf ("Target Output (AB): ");
    for (int i = 0; i < len_output; i++)
    {
        if (target_output[i][0] == 0)
        {
            printf ("[LEFT][%d], ", target_output[i][1]);
        }
        else
        {
            printf ("[RIGHT][%d], ", target_output[i][1]);
        }
    }
    printf ("\n");

    printf ("Target Output (BA): ");
    for (int i = 0; i < len_output; i++)
    {
        if (target_output_swapped[i][0] == 0)
        {
            printf ("[LEFT][%d], ", target_output_swapped[i][1]);
        }
        else
        {
            printf ("[RIGHT][%d], ", target_output_swapped[i][1]);
        }
    }
    printf ("\n");
    printf ("==========================================================================================\n");
#endif

    //
    //  Input-Left: Default[0, 1, 2, 3] >>> [0, 1, 3, 2], [0, 2, 1, 3], [0, 2, 3, 1]
    //    
    //  idx_perms_ext_input_left, idx_perms_ext_input_right, idx_perms_intenral
    int perm_offset = 0;
    int result_output[len_output][2];
    int result_output_swapped[len_output][2];
    int idx_perms_output[len_output];
    int idx_perms_output_swapped[len_output];
    int idx_perms_input_left[len_input_left];
    int idx_perms_input_right[len_input_right];
    
    //
    info_config* list_configurations = (info_config*)malloc(sizeof(info_config) * num_perms_trans * SIZE_SWAP_INPUTS);
    
    //
    //  |Internal Indices|!
    //
    for (int perm_internal = 0; perm_internal < num_perms_int_input_left; perm_internal++)
    {
        // 2 Types: (Ext., Int.) and (Int., Ext.) 
        for (int perm_int_input_left = 0; perm_int_input_left < 2; perm_int_input_left++)
        {
            // |External Indices in Input-Left|!
            for (int perm_ext_input_left = 0; perm_ext_input_left < num_perms_ext_input_left; perm_ext_input_left++)
            {
                // 2 Types: (Ext., Int.) and (Int., Ext.)
                for (int perm_int_input_right = 0; perm_int_input_right < 2; perm_int_input_right++)
                {
                    // |External Indices in Input-Right|!
                    for (int perm_ext_input_right = 0; perm_ext_input_right < num_perms_ext_input_right; perm_ext_input_right++)
                    {
                        //
                        //  Result Output from (Transposed) Inputs
                        //
                        //  LEFT
                        for (int i = 0; i < num_ext_input_left; i++)
                        {
                            result_output[i][0] = 0;
                            result_output[i][1] = idx_perms_ext_input_left[perm_ext_input_left * num_ext_input_left + i];
                            // printf ("L) result_output[%d] = (%d,%d)\n", i, 0, result_output[i][1]);
                        }

                        //  RIGHT
                        for (int i = 0; i < num_ext_input_right; i++)
                        {
                            result_output[i + num_ext_input_left][0] = 1;
                            result_output[i + num_ext_input_left][1] = idx_perms_ext_input_right[perm_ext_input_right * num_ext_input_right + i];
                            // printf ("R) result_output[%d] = (%d,%d)\n", i, 1, result_output[i][1]);
                        }

                        //  RIGHT
                        for (int i = 0; i < num_ext_input_left; i++)
                        {
                            result_output_swapped[i + num_ext_input_right][0] = 1;
                            // result_output_swapped[i + num_ext_input_right][0] = 0;
                            result_output_swapped[i + num_ext_input_right][1] = idx_perms_ext_input_left[perm_ext_input_left * num_ext_input_left + i];
                        }

                        // LEFT
                        for (int i = 0; i < num_ext_input_right; i++)
                        {
                            result_output_swapped[i][0] = 0;
                            // result_output_swapped[i][0] = 1;
                            result_output_swapped[i][1] = idx_perms_ext_input_right[perm_ext_input_right * num_ext_input_right + i];
                        }


                    #ifdef DEBUG_PERMUTATIONS_DETAIL
                        for (int i = 0; i < num_ext_input_left + num_ext_input_right; i++)
                        {
                            if (i < num_ext_input_right)
                                printf ("[%d][R(%d)][%d], ", i, result_output_swapped[i][0], result_output_swapped[i][1]);
                            else
                                printf ("[%d][L(%d)][%d], ", i, result_output_swapped[i][0], result_output_swapped[i][1]);
                        }
                        printf (" >>> ");
                        for (int i = 0; i < len_output; i++)
                        {
                            if (target_output[i][0] == 0)
                                printf ("[L(%d)][%d], ", target_output_swapped[i][0], target_output_swapped[i][1]);
                            else
                                printf ("[R(%d)][%d], ", target_output_swapped[i][0], target_output_swapped[i][1]);
                        }
                        printf ("\n");
                    #endif

                    #ifdef DEBUG_PERMUTATIONS_DETAIL
                        for (int i = 0; i < num_ext_input_left + num_ext_input_right; i++)
                        {
                            if (i < num_ext_input_left)
                                printf ("[%d][LEFT(%d)][%d], ", i, result_output[i][0], result_output[i][1]);
                            else
                                printf ("[%d][RIGHT(%d)][%d], ", i, result_output[i][0], result_output[i][1]);
                        }
                        printf (" >>> ");
                        for (int i = 0; i < len_output; i++)
                        {
                            if (target_output[i][0] == 0)
                                printf ("[LEFT(%d)][%d], ", target_output[i][0], target_output[i][1]);
                            else
                                printf ("[RIGHT(%d)][%d], ", target_output[i][0], target_output[i][1]);
                        }
                    #endif

                        //
                        //  The result output [0,1,2,...,n]  should be transposed to the target output.
                        //
                        for (int i = 0; i < len_output; i++)
                        {
                            for (int j = 0; j < len_output; j++)
                            {
                                if (result_output[j][0] == target_output[i][0] && result_output[j][1] == target_output[i][1])
                                {
                                    idx_perms_output[i] = j;
                                }
                            }
                        }

                        //
                        for (int i = 0; i < len_output; i++)
                        {
                            for (int j = 0; j < len_output; j++)
                            {
                                if (result_output_swapped[j][0] == target_output_swapped[i][0] && result_output_swapped[j][1] == target_output_swapped[i][1])
                                {
                                    idx_perms_output_swapped[i] = j;
                                }
                            }
                        }

                        //
                        //  If 0,1,2,3,..,n, Then, Tensor-Transposition is not needed.
                        //
                        list_configurations[perm_offset].info_dims_output       = (int*)malloc(sizeof(int) * len_output);
                        list_configurations[perm_offset].info_perms_output      = (int*)malloc(sizeof(int) * len_output);
                        list_configurations[perm_offset].info_perms_input_left  = (int*)malloc(sizeof(int) * len_input_left);
                        list_configurations[perm_offset].info_perms_input_right = (int*)malloc(sizeof(int) * len_input_right);

                        list_configurations[perm_offset + num_perms_trans].info_dims_output         = (int*)malloc(sizeof(int) * len_output);
                        list_configurations[perm_offset + num_perms_trans].info_perms_output        = (int*)malloc(sizeof(int) * len_output);
                        list_configurations[perm_offset + num_perms_trans].info_perms_input_left    = (int*)malloc(sizeof(int) * len_input_right);
                        list_configurations[perm_offset + num_perms_trans].info_perms_input_right   = (int*)malloc(sizeof(int) * len_input_left);


                        //  Copy The Output-Permutation
                        for (int i = 0; i < len_output; i++)
                        {
                            // AB
                            list_configurations[perm_offset].info_perms_output[i]                   = idx_perms_output[i];
                            // printf ("%d(%d) ", idx_perms_output[i], info_tc->info_output[idx_perms_output[i]].size);
                            list_configurations[perm_offset].info_dims_output[i]                    = info_tc->info_output[idx_perms_output[i]].size;
                            
                            // BA
                            list_configurations[perm_offset + num_perms_trans].info_perms_output[i] = idx_perms_output_swapped[i];
                            // printf ("%d(%d) ", idx_perms_output_swapped[i], info_tc->info_output[idx_perms_output_swapped[i]].size);
                            list_configurations[perm_offset + num_perms_trans].info_dims_output[i]  = info_tc->info_output[idx_perms_output_swapped[i]].size;
                        }
                        // printf ("\n");
                    
                    #ifdef DEBUG_PERMUTATIONS_DETAIL_INNER
                        printf ("[%4d] Output's Perm.: ", perm_offset);
                        for (int i = 0; i < len_output; i++)
                        {
                            printf ("%d [%d], ", idx_perms_output[i], list_configurations[perm_offset].info_perms_output[i]);
                        }
                        printf ("[%4d] (Swapped) Output's Perm.: ", perm_offset + num_perms_trans);
                        for (int i = 0; i < len_output; i++)
                        {
                            printf ("%d [%d], ", idx_perms_output_swapped[i], list_configurations[perm_offset + num_perms_trans].info_perms_output[i]);
                        }
                        printf ("\n");
                    #endif

                        // 
                        //  Need to check an overall cost for each case.
                        //
                        if (perm_int_input_left == 0)       // Left-0 : (Ext., Int.)
                        {
                            //  [LEFT] (0) Ext.
                            for (int i = 0; i < num_ext_input_left; i++)
                            {
                                idx_perms_input_left[i]                         = idx_perms_ext_input_left[perm_ext_input_left * num_ext_input_left + i];
                            }
                            //  [Left] (1) Int.
                            for (int i = 0; i < num_int_input_left; i++)
                            {
                                idx_perms_input_left[num_ext_input_left + i]    = idx_perms_int_input_left[perm_internal * num_int_input_left + i];
                            }

                            //  Copy The Input-LEFT-Permutation
                            for (int i = 0; i < len_input_left; i++)
                            {
                                list_configurations[perm_offset].info_perms_input_left[i]                       = idx_perms_input_left[i];
                                list_configurations[perm_offset + num_perms_trans].info_perms_input_right[i]    = idx_perms_input_left[i];
                            }

                            //
                            //
                            //
                            if (perm_int_input_right == 0)  // Right-0: (Ext., Int.)
                            {
                                //  [RIGHT] (0) Ext.
                                for (int i = 0; i < num_ext_input_right; i++)
                                {
                                    idx_perms_input_right[i]                        = idx_perms_ext_input_right[perm_ext_input_right * num_ext_input_right + i];
                                }
                                //  [RIGHT] (1) Int.
                                for (int i = 0; i < num_int_input_right; i++)
                                {
                                    idx_perms_input_right[num_ext_input_right + i]  = idx_perms_int_input_right[perm_internal * num_int_input_right + i];
                                }

                                //  Copy the Input-RIGHT-Permutation
                                for(int i= 0; i < len_input_right; i++)
                                {
                                    list_configurations[perm_offset].info_perms_input_right[i]                  = idx_perms_input_right[i];
                                    list_configurations[perm_offset + num_perms_trans].info_perms_input_left[i] = idx_perms_input_right[i];
                                }
                                
                            #ifdef DEBUG_PERMUTATIONS_DETAIL_INNER
                                printf ("[#1] LEFT: ");
                                for (int k = 0; k < len_input_left; k++)
                                {
                                    printf ("%d [%d], ", idx_perms_input_left[k], list_configurations[perm_offset].info_perms_input_left[k]);
                                }
                                printf (" && ");

                                printf ("RIGHT: ");
                                for (int k = 0; k < len_input_right; k++)
                                {
                                    printf ("%d [%d], ", idx_perms_input_right[k], list_configurations[perm_offset].info_perms_input_right[k]);
                                }
                                printf ("\n");


                                printf ("[#0] LEFT: ");
                                for (int k = 0; k < len_input_left; k++)
                                {
                                    printf ("%d [%d], ", idx_perms_input_left[k], list_configurations[perm_offset + num_perms_trans].info_perms_input_right[k]);
                                }
                                printf (" && ");

                                printf ("RIGHT: ");
                                for (int k = 0; k < len_input_right; k++)
                                {
                                    printf ("%d [%d], ", idx_perms_input_right[k], list_configurations[perm_offset + num_perms_trans].info_perms_input_left[k]);
                                }
                                printf ("\n");
                            #endif

                                perm_offset++;
                            }
                            else                            // Right-1: (Int., Ext.)
                            {
                                //  [RIGHT] (0) Int.
                                for (int i = 0; i < num_int_input_right; i++)
                                {
                                    idx_perms_input_right[i]                        = idx_perms_int_input_right[perm_internal * num_int_input_right + i];
                                }
                                //  [RIGHT] (1) Ext.
                                for (int i = 0; i < num_ext_input_right; i++)
                                {
                                    idx_perms_input_right[num_int_input_right + i]  = idx_perms_ext_input_right[perm_ext_input_right * num_ext_input_right + i];
                                }

                                //  Copy the Input-RIGHT-Permutation
                                for(int i= 0; i < len_input_right; i++)
                                {
                                    list_configurations[perm_offset].info_perms_input_right[i]                  = idx_perms_input_right[i];
                                    list_configurations[perm_offset + num_perms_trans].info_perms_input_left[i] = idx_perms_input_right[i];
                                }

                            #ifdef DEBUG_PERMUTATIONS_DETAIL_INNER
                                printf ("[#2-1] LEFT: ");
                                for (int k = 0; k < len_input_left; k++)
                                {
                                    printf ("%d [%d], ", idx_perms_input_left[k], list_configurations[perm_offset].info_perms_input_left[k]);
                                }
                                printf (" && ");

                                printf ("RIGHT: ");
                                for (int k = 0; k < len_input_right; k++)
                                {
                                    printf ("%d [%d], ", idx_perms_input_right[k], list_configurations[perm_offset].info_perms_input_right[k]);
                                }
                                printf ("\n");


                                printf ("[#2-2] LEFT: ");
                                for (int k = 0; k < len_input_left; k++)
                                {
                                    printf ("%d [%d], ", idx_perms_input_left[k], list_configurations[perm_offset + num_perms_trans].info_perms_input_right[k]);
                                }
                                printf (" && ");

                                printf ("RIGHT: ");
                                for (int k = 0; k < len_input_right; k++)
                                {
                                    printf ("%d [%d], ", idx_perms_input_right[k], list_configurations[perm_offset + num_perms_trans].info_perms_input_left[k]);
                                }
                                printf ("\n");
                            #endif
                                perm_offset++;
                            }
                        }
                        else                                // Left-1 : (Int., Ext.)
                        {
                            //  [LEFT] (0) Int.
                            for (int i = 0; i < num_int_input_left; i++)
                            {
                                idx_perms_input_left[i]                         = idx_perms_int_input_left[perm_internal * num_int_input_left + i];
                            }
                            //  [LEFT] (1) Ext.
                            for (int i = 0; i < num_ext_input_left; i++)
                            {
                                idx_perms_input_left[num_int_input_left + i]    = idx_perms_ext_input_left[perm_ext_input_left * num_ext_input_left + i];
                            }

                            //  Copy The Input-LEFT-Permutation
                            for (int i = 0; i < len_input_left; i++)
                            {
                                list_configurations[perm_offset].info_perms_input_left[i]                       = idx_perms_input_left[i];
                                list_configurations[perm_offset + num_perms_trans].info_perms_input_right[i]    = idx_perms_input_left[i];
                            }

                            //
                            //
                            //
                            if (perm_int_input_right == 0)  // Right-0: (Ext., Int.)
                            {
                                //  [RIGHT] (0) Ext.
                                for (int i = 0; i < num_ext_input_right; i++)
                                {
                                    idx_perms_input_right[i]                        = idx_perms_ext_input_right[perm_ext_input_right * num_ext_input_right + i];
                                }
                                //  [RIGHT] (1) Int.
                                for (int i = 0; i < num_int_input_right; i++)
                                {
                                    idx_perms_input_right[num_ext_input_right + i]  = idx_perms_int_input_right[perm_internal * num_int_input_right + i];
                                }

                                //  Copy the Input-RIGHT-Permutation
                                for(int i= 0; i < len_input_right; i++)
                                {
                                    list_configurations[perm_offset].info_perms_input_right[i]                  = idx_perms_input_right[i];
                                    list_configurations[perm_offset + num_perms_trans].info_perms_input_left[i] = idx_perms_input_right[i];
                                }

                            #ifdef DEBUG_PERMUTATIONS_DETAIL_INNER
                                printf ("[#3-1] LEFT: ");
                                for (int k = 0; k < len_input_left; k++)
                                {
                                    printf ("%d [%d], ", idx_perms_input_left[k], list_configurations[perm_offset].info_perms_input_left[k]);
                                }
                                printf (" && ");

                                printf ("RIGHT: ");
                                for (int k = 0; k < len_input_right; k++)
                                {
                                    printf ("%d [%d], ", idx_perms_input_right[k], list_configurations[perm_offset].info_perms_input_right[k]);
                                }
                                printf ("\n");

                                printf ("[#3-2] LEFT: ");
                                for (int k = 0; k < len_input_left; k++)
                                {
                                    printf ("%d [%d], ", idx_perms_input_left[k], list_configurations[perm_offset + num_perms_trans].info_perms_input_right[k]);
                                }
                                printf (" && ");

                                printf ("RIGHT: ");
                                for (int k = 0; k < len_input_right; k++)
                                {
                                    printf ("%d [%d], ", idx_perms_input_right[k], list_configurations[perm_offset + num_perms_trans].info_perms_input_left[k]);
                                }
                                printf ("\n");
                            #endif
                                perm_offset++;
                            }
                            else                            // Right-1: (Int., Ext.)
                            {
                                //  [RIGHT] (0) Int.
                                for (int i = 0; i < num_int_input_right; i++)
                                {
                                    idx_perms_input_right[i]                        = idx_perms_int_input_right[perm_internal * num_int_input_right + i];
                                }
                                //  [RIGHT] (1) Ext.
                                for (int i = 0; i < num_ext_input_right; i++)
                                {
                                    idx_perms_input_right[num_int_input_right + i]  = idx_perms_ext_input_right[perm_ext_input_right * num_ext_input_right + i];
                                }

                                //  Copy the Input-RIGHT-Permutation
                                for(int i= 0; i < len_input_right; i++)
                                {
                                    list_configurations[perm_offset].info_perms_input_right[i]                      = idx_perms_input_right[i];
                                    list_configurations[perm_offset + num_perms_trans].info_perms_input_left[i]     = idx_perms_input_right[i];
                                }

                            #ifdef DEBUG_PERMUTATIONS_DETAIL_INNER
                                printf ("[#4-1] LEFT: ");
                                for (int k = 0; k < len_input_left; k++)
                                {
                                    printf ("%d [%d], ", idx_perms_input_left[k], list_configurations[perm_offset].info_perms_input_left[k]);
                                }
                                printf (" && ");

                                printf ("RIGHT: ");
                                for (int k = 0; k < len_input_right; k++)
                                {
                                    printf ("%d [%d], ", idx_perms_input_right[k], list_configurations[perm_offset].info_perms_input_right[k]);
                                }
                                printf ("\n");

                                printf ("[#4-2] LEFT: ");
                                for (int k = 0; k < len_input_left; k++)
                                {
                                    printf ("%d [%d], ", idx_perms_input_left[k], list_configurations[perm_offset + num_perms_trans].info_perms_input_right[k]);
                                }
                                printf (" && ");

                                printf ("RIGHT: ");
                                for (int k = 0; k < len_input_right; k++)
                                {
                                    printf ("%d [%d], ", idx_perms_input_right[k], list_configurations[perm_offset + num_perms_trans].info_perms_input_left[k]);
                                }
                                printf ("\n");
                            #endif
                                perm_offset++;
                            }
                        }
                    }
                }
            }
        }
    }

    //
    free(idx_perms_ext_input_left);
    free(idx_perms_int_input_left);
    free(idx_perms_ext_input_right);
    free(idx_perms_int_input_right);

    //
    //  Simple Categorizations
    //
    // check_simple_four_cases(num_perms_trans, list_configurations, info_tc);
#ifdef DEBUG_PERMUTATIONS_ENUMERATE
    for (int i = 0; i < (num_perms_trans * SIZE_SWAP_INPUTS); i++)
    {
        printf ("[%3d] L: |", i);
        
        if (i < num_perms_trans)
        {
            for (int j = 0; j < len_input_left; j++)
            {
                printf ("%d ", list_configurations[i].info_perms_input_left[j]);
            }
        }
        else
        {
            for (int j = 0; j < len_input_right; j++)
            {
                printf ("%d ", list_configurations[i].info_perms_input_left[j]);
            }
        }
        printf ("| * R: |");

        if (i < num_perms_trans)
        {
            for (int j = 0; j < len_input_right; j++)
            {
                printf ("%d ", list_configurations[i].info_perms_input_right[j]);
            }
        }
        else
        {
            for (int j = 0; j < len_input_left; j++)
            {
                printf ("%d ", list_configurations[i].info_perms_input_right[j]);
            }
        }
        printf ("| = C: |");

        for (int j = 0; j < len_output; j++)
        {
            printf ("%d ", list_configurations[i].info_perms_output[j]);
        }
        printf ("|\n");
    }

#endif

    //
    //  Model for All Configurations
    //
    printf ("===========================================================================\n");
    printf (">>> opt_manual: %d\n", opt_manual);
    int picked_config = model_base(num_perms_trans, list_configurations, info_tc);
    // int picked_config = 0;
    if (opt_manual != -1)
    {
        printf (" Picked Configuration #: %d (But not actually)\n", picked_config);
    }
    else
    {
        printf (" Picked Configuration #: %d\n", picked_config);
    }    
    printf ("===========================================================================\n");

    //
    //  To Create The Best Configuration based on the Model or to Pick a Configuration According to the input.
    //
    int default_configuration;
    if (opt_manual != -1)
    {
        if (target_configuration < (num_perms_trans * 2))
            default_configuration = target_configuration;
        else
            default_configuration = 0;
    }
    else
    {
        default_configuration = picked_config;
    }    

    printf ("[%s] Picked Configuration #: %d among %d * 2 (from 0 to %d)\n", __func__, default_configuration, num_perms_trans, num_perms_trans * 2 - 1);

#ifdef DEBUG_PERMUTATIONS_RESULT
    // Output
    printf ("[%s][Permutation Info.][Output] ", __func__);
    for (int i = 0; i < len_output; i++)
    {
        printf ("%d, ", list_configurations[default_configuration].info_perms_output[i]);
    }
    printf ("\n");

    if (default_configuration < num_perms_trans)
    {
        printf ("[%s][Permutation Info.][Normal]\n", __func__);
        // Input-Left
        printf ("[%s][Permutation Info.][Input][LEFT] ", __func__);
        for (int i = 0; i < len_input_left; i++)
        {
            printf ("%d, ", list_configurations[default_configuration].info_perms_input_left[i]);
        }
        printf ("\n");

        // Input-Right
        printf ("[%s][Permutation Info.][Input][RIGHT] ", __func__);
        for (int i = 0; i < len_input_right; i++)
        {
            printf ("%d, ", list_configurations[default_configuration].info_perms_input_right[i]);
        }
    }
    else
    {
        printf ("[%s][Permutation Info.][Swapped]\n", __func__);
        // Input-Left
        printf ("[%s][Permutation Info.][Input][RIGHT] ", __func__);
        for (int i = 0; i < len_input_left; i++)
        {
            printf ("%d, ", list_configurations[default_configuration].info_perms_input_right[i]);
        }
        printf ("\n");

        // Input-Right
        printf ("[%s][Permutation Info.][Input][LEFT] ", __func__);
        for (int i = 0; i < len_input_right; i++)
        {
            printf ("%d, ", list_configurations[default_configuration].info_perms_input_left[i]);
        }
    }
    printf ("\n");
#endif

    //
    //  This is for the Picked Configuration
    //
    result_tt_output        = (tt*)malloc(sizeof(tt));
    result_tt_input_left    = (tt*)malloc(sizeof(tt));
    result_tt_input_right   = (tt*)malloc(sizeof(tt));

    //  DEFAULTS
    result_tt_output->transpose             = OPT_TRANSPOSE_NO;
    result_tt_output->dgemm_transpose       = OPT_TRANSPOSE_NO;     // Useless
    result_tt_input_left->transpose         = OPT_TRANSPOSE_NO;
    result_tt_input_right->transpose        = OPT_TRANSPOSE_NO;

    for (int i = 0; i < len_output; i++)
    {
        if (list_configurations[default_configuration].info_perms_output[i] != i)
        {
            printf ("[%s][Output] Transposition is Required\n", __func__);
            result_tt_output->transpose         = OPT_TRANSPOSE_YES;
            break;
        }
    }

    if (default_configuration < num_perms_trans)
    {
        //
        //  Input-Tensor (LEFT)
        //
        for (int i = 0; i < len_input_left; i++)
        {
            if (list_configurations[default_configuration].info_perms_input_left[i] != i)
            {
                printf ("[%s][Input][Left] Transposition is Required\n", __func__);
                result_tt_input_left->transpose     = OPT_TRANSPOSE_YES;
                break;
            }
        }
        
        //
        if (find_size(info_input_left[list_configurations[default_configuration].info_perms_input_left[0]].name, problem_size_internal, len_internal_indices) == 0)
        {
            //  The FVI of t2' is not an Internal Index, resulting in (E_A, K) Form (N)
            result_tt_input_left->dgemm_transpose   = OPT_TRANSPOSE_NO;     // (default) N
        }
        else
        {
            //  The FVI of t2' is an Internal Index, resulting in (K, E_A) Form (T)
            result_tt_input_left->dgemm_transpose   = OPT_TRANSPOSE_YES;
        }
        

        //
        //  Input-Tensor (RIGHT)
        //
        for (int i = 0; i < len_input_right; i++)
        {
            if (list_configurations[default_configuration].info_perms_input_right[i] != i)
            {
                printf ("[%s][Input][Right] Transposition is Required\n", __func__);
                result_tt_input_right->transpose = OPT_TRANSPOSE_YES;
                break;
            }
        }
        
        //
        if (find_size(info_input_right[list_configurations[default_configuration].info_perms_input_right[0]].name, problem_size_internal, len_internal_indices) == 0)
        {
            //  The FVI of v2' is not an Internal Index, resulting in (E_B, K) Form (T)
            result_tt_input_right->dgemm_transpose  = OPT_TRANSPOSE_YES;    // (default) T
        }
        else
        {
            //  The FVI of v2' is an Internal Index, resulting in (K, E_B) Form (N)
            result_tt_input_right->dgemm_transpose  = OPT_TRANSPOSE_NO;
        }
    }
    else
    {
        //
        //  Input-Tensor (LEFT)
        //
        for (int i = 0; i < len_input_left; i++)
        {
            if (list_configurations[default_configuration].info_perms_input_right[i] != i)
            {
                printf ("[%s][Input][RIGHT] Transposition is Required\n", __func__);
                result_tt_input_right->transpose         = OPT_TRANSPOSE_YES;
                break;
            }
        }
        
        //
        if (find_size(info_input_right[list_configurations[default_configuration].info_perms_input_left[0]].name, problem_size_internal, len_internal_indices) == 0)
        {
            //  The FVI of t2' is not an Internal Index, resulting in (E_A, K) Form (N)
            result_tt_input_left->dgemm_transpose = OPT_TRANSPOSE_NO;     // (default) N
        }
        else
        {
            //  The FVI of t2' is an Internal Index, resulting in (K, E_A) Form (T)
            result_tt_input_left->dgemm_transpose = OPT_TRANSPOSE_YES;
        }
        

        //
        //  Input-Tensor (RIGHT)
        //
        for (int i = 0; i < len_input_right; i++)
        {
            if (list_configurations[default_configuration].info_perms_input_left[i] != i)
            {
                printf ("[%s][Input][Left] Transposition is Required\n", __func__);
                result_tt_input_left->transpose = OPT_TRANSPOSE_YES;
                break;
            }
        }
        
        //
        if (find_size(info_input_left[list_configurations[default_configuration].info_perms_input_right[0]].name, problem_size_internal, len_internal_indices) == 0)
        {
            //  The FVI of v2' is not an Internal Index, resulting in (E_B, K) Form (T)
            result_tt_input_right->dgemm_transpose  = OPT_TRANSPOSE_YES;    // (default) T
        }
        else
        {
            //  The FVI of v2' is an Internal Index, resulting in (K, E_B) Form (N)
            result_tt_input_right->dgemm_transpose  = OPT_TRANSPOSE_NO;
        }
    }

    //
    if (default_configuration < num_perms_trans)
    {   
        result_tt_output->len_indice        = len_output;
        result_tt_input_left->len_indice    = len_input_left;
        result_tt_input_right->len_indice   = len_input_right;

        result_tt_output->ttlg_dims         = (int*)malloc(sizeof(int) * len_output);
        result_tt_output->ttlg_perms        = (int*)malloc(sizeof(int) * len_output);

        result_tt_input_left->ttlg_dims     = (int*)malloc(sizeof(int) * len_input_left);
        result_tt_input_left->ttlg_perms    = (int*)malloc(sizeof(int) * len_input_left);

        result_tt_input_right->ttlg_dims    = (int*)malloc(sizeof(int) * len_input_right);
        result_tt_input_right->ttlg_perms   = (int*)malloc(sizeof(int) * len_input_right);
        
        //
        //  [To-Do] Should be Fixed 
        //
        for (int i = 0; i < len_output; i++)
        {
            result_tt_output->ttlg_perms[i]                                                                 = list_configurations[default_configuration].info_perms_output[i];
            result_tt_output->ttlg_dims[list_configurations[default_configuration].info_perms_output[i]]    = find_size(info_output[i].name, problem_size_external, len_external_indices);
        }
        
        //
        for (int i = 0; i < len_input_left; i++)
        {
            int tmp_dim = find_size(info_input_left[i].name, problem_size_external, len_external_indices);
            if (tmp_dim > 0)
            {
                result_tt_input_left->ttlg_dims[i]      = tmp_dim;
            }
            else
            {
                result_tt_input_left->ttlg_dims[i]      = find_size(info_input_left[i].name, problem_size_internal, len_internal_indices);
            }
            
            result_tt_input_left->ttlg_perms[i]     = list_configurations[default_configuration].info_perms_input_left[i];
        #ifdef DEBUG_MAKE_INFO_TTLG
            printf ("info_input_left[%d][%s][%d] vs result_tt_input_left[%d][%d]\n", i, info_input_left[i].name, info_input_left[i].size, i, result_tt_input_left->ttlg_dims[i]);   
        #endif
        }

        //
        for (int i = 0; i < len_input_right; i++)
        {
            int tmp_dim = find_size(info_input_right[i].name, problem_size_external, len_external_indices);
            if (tmp_dim > 0)
            {
                result_tt_input_right->ttlg_dims[i]     = tmp_dim;
            }
            else
            {
                result_tt_input_right->ttlg_dims[i]     = find_size(info_input_right[i].name, problem_size_internal, len_internal_indices);
            }
            
            result_tt_input_right->ttlg_perms[i]    = list_configurations[default_configuration].info_perms_input_right[i];
        #ifdef DEBUG_MAKE_INFO_TTLG
            printf ("info_input_right[%d][%s][%d] vs result_tt_input_right[%d][%d]\n", i, info_input_right[i].name, info_input_right[i].size, i, result_tt_input_right->ttlg_dims[i]);
        #endif
        }
    }
    else
    {
        result_tt_output->len_indice        = len_output;
        result_tt_input_left->len_indice    = len_input_right;
        result_tt_input_right->len_indice   = len_input_left;

        result_tt_output->ttlg_dims         = (int*)malloc(sizeof(int) * len_output);
        result_tt_output->ttlg_perms        = (int*)malloc(sizeof(int) * len_output);

        result_tt_input_left->ttlg_dims     = (int*)malloc(sizeof(int) * len_input_right);
        result_tt_input_left->ttlg_perms    = (int*)malloc(sizeof(int) * len_input_right);

        result_tt_input_right->ttlg_dims    = (int*)malloc(sizeof(int) * len_input_left);
        result_tt_input_right->ttlg_perms   = (int*)malloc(sizeof(int) * len_input_left);
        
        // BA
        // list_configurations[perm_offset + num_perms_trans].info_perms_output[i] = idx_perms_output_swapped[i];

        //
        //  [To-Do] Should be Fixed 
        //
        // printf (">>> default_configuraiton: %d\n", default_configuration);
        for (int i = 0; i < len_output; i++)
        {
            result_tt_output->ttlg_perms[i]                                                                 = list_configurations[default_configuration].info_perms_output[i];
            result_tt_output->ttlg_dims[list_configurations[default_configuration].info_perms_output[i]]    = find_size(info_output[i].name, problem_size_external, len_external_indices);
        }
        
        for (int i = 0; i < len_input_right; i++)
        {
            int tmp_dim = find_size(info_input_right[i].name, problem_size_external, len_external_indices);
            if (tmp_dim > 0)
            {
                result_tt_input_left->ttlg_dims[i]      = tmp_dim;
            }
            else
            {
                result_tt_input_left->ttlg_dims[i]      = find_size(info_input_right[i].name, problem_size_internal, len_internal_indices);
            }
            
            result_tt_input_left->ttlg_perms[i]     = list_configurations[default_configuration].info_perms_input_left[i];
        #ifdef DEBUG_MAKE_INFO_TTLG
            printf ("info_input_left[%d][%s][%d] vs result_tt_input_left[%d][%d]\n", i, info_input_right[i].name, info_input_right[i].size, i, result_tt_input_left->ttlg_dims[i]);   
        #endif
        }

        for (int i = 0; i < len_input_left; i++)
        {
            int tmp_dim = find_size(info_input_left[i].name, problem_size_external, len_external_indices);
            if (tmp_dim > 0)
            {
                result_tt_input_right->ttlg_dims[i]     = tmp_dim;
            }
            else
            {
                result_tt_input_right->ttlg_dims[i]     = find_size(info_input_left[i].name, problem_size_internal, len_internal_indices);
            }
            
            result_tt_input_right->ttlg_perms[i]    = list_configurations[default_configuration].info_perms_input_right[i];
        #ifdef DEBUG_MAKE_INFO_TTLG
            printf ("info_input_right[%d][%s][%d] vs result_tt_input_right[%d][%d]\n", i, info_input_left[i].name, info_input_left[i].size, i, result_tt_input_right->ttlg_dims[i]);
        #endif
        }
    }


    if (default_configuration < num_perms_trans)
        return TYPE_AB;
    else
        return TYPE_BA;
    //printf ("[%s]======================================================================\n", __func__);
}