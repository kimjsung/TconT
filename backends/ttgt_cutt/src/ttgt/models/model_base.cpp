#include "../ttgt_ttlg.h"

//#define DEBUG_PERMUTATIONS
//#define DEBUG_PERMUTATIONS_DETAIL
//#define DEBUG_PERMUTATIONS_DETAIL_INNER
//#define DEBUG_TILES
//#define DEBUG_MAKE_INFO_TTLG


/*
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
    int     transpose_input_left;
    int     transpose_input_right;
    int     transpose_output;
    int*    info_perms_input_left;
    int*    info_perms_input_right;
    int*    info_perms_output;
}info_config;
*/

//
int model_base(int num_configurations, info_config* list_configurations, tc* info_tc)
{
    //
    printf ("[%s] Figuring out %d configurations' cost\n", __func__, num_configurations);

    //
    int     best_config_idx_AB  = 0;
    int     best_config_idx_BA  = 0;
    double  min_total_cost_AB   = 1000000000.0;
    double  min_total_cost_BA   = 1000000000.0;
    double  picked_cost_trans_A = 0.0;
    double  picked_cost_trans_B = 0.0;
    double  picked_cost_trans_C = 0.0;

    int  len_output         = info_tc->len_output;
    int  len_input_left     = info_tc->len_input_left;
    int  len_input_right    = info_tc->len_input_right;
    idx* info_output        = info_tc->info_output;
    idx* info_input_left    = info_tc->info_input_left;
    idx* info_input_right   = info_tc->info_input_right;

    int* tmp_tt_output_ttlg_dims        = (int*)malloc(sizeof(int) * (len_output));
    int* tmp_tt_output_ttlg_perms       = (int*)malloc(sizeof(int) * (len_output));
    int* tmp_tt_input_left_ttlg_dims    = (int*)malloc(sizeof(int) * (len_input_left));
    int* tmp_tt_input_left_ttlg_perms   = (int*)malloc(sizeof(int) * (len_input_left));
    int* tmp_tt_input_right_ttlg_dims   = (int*)malloc(sizeof(int) * (len_input_right));
    int* tmp_tt_input_right_ttlg_perms  = (int*)malloc(sizeof(int) * (len_input_right));

    //  dims (might be common.)
    for (int i = 0; i < len_output; i++)        tmp_tt_output_ttlg_dims[i] = info_output[i].size;
    for (int i = 0; i < len_input_left; i++)    tmp_tt_input_left_ttlg_dims[i] = info_input_left[i].size;
    for (int i = 0; i < len_input_right; i++)   tmp_tt_input_right_ttlg_dims[i] = info_input_right[i].size;

    //
    //  printf (":: Estimated BW: %lf\n", 1000 * ttlg_transpose_time(len_input_left, ttlg_dims_input_left, ttlg_perms_input_left));
    //
    for (int i = 0; i < num_configurations; i++)
    {
        int     transpose_input_left    = OPT_TRANSPOSE_NO;
        int     transpose_input_right   = OPT_TRANSPOSE_NO;
        int     transpose_output        = OPT_TRANSPOSE_NO;
        double  cost_transpositions     = 0.0;
        double  cost_trans_A            = 0.0;
        double  cost_trans_B            = 0.0;
        double  cost_trans_C            = 0.0;


        //  INPUT-LEFT
        for (int j = 0; j < len_input_left; j++)
        {
            if (list_configurations[i].info_perms_input_left[j] != j)
            {
                transpose_input_left = OPT_TRANSPOSE_YES;
                break;
            }
        }

        //  INPUT-RIGHT
        for (int j = 0; j < len_input_right; j++)
        {
            if (list_configurations[i].info_perms_input_right[j] != j)
            {
                transpose_input_right = OPT_TRANSPOSE_YES;
                break;
            }
        }

        //  OUTPUT
        for (int j = 0; j < len_output; j++)
        {
            if (list_configurations[i].info_perms_output[j] != j)
            {
                transpose_output = OPT_TRANSPOSE_YES;
                break;
            }
        }

        // printf ("====================================================================================================================================\n");
        //  INPUT-LEFT
        if (transpose_input_left == OPT_TRANSPOSE_YES)
        {
            // cost_trans_A = ttlg_transpose_time(len_input_left, tmp_tt_input_left_ttlg_dims, list_configurations[i].info_perms_input_left);
           // for (int j = 0; j < 4; j++)
           //     printf ("%d ", list_configurations[i].info_perms_input_left[j]);
            // printf ("[%3d] >>> Left: %f\n", i, tmp);
            cost_transpositions += cost_trans_A;
            // cost_transpositions += 1000 * ttlg_transpose_time(len_input_left, tmp_tt_input_left_ttlg_dims, list_configurations[i].info_perms_input_left);
        }
       // printf (" - ");
        //  INPUT-RIGHT
        if (transpose_input_right == OPT_TRANSPOSE_YES)
        {
            // cost_trans_B = ttlg_transpose_time(len_input_right, tmp_tt_input_right_ttlg_dims, list_configurations[i].info_perms_input_right);
            // printf ("[%3d] >>> Right: %f\n", i, tmp);
            //for (int j = 0; j < 4; j++)
            //    printf ("%d ", list_configurations[i].info_perms_input_right[j]);
            cost_transpositions += cost_trans_B;
            // cost_transpositions += 1000 * ttlg_transpose_time(len_input_right, tmp_tt_input_right_ttlg_dims, list_configurations[i].info_perms_input_right);
        }

        //  DGEMM
        //printf (" - ");

        //  OUTPUT
        if (transpose_output == OPT_TRANSPOSE_YES)
        {
            // cost_trans_C = 1000 * ttlg_transpose_time(len_output, tmp_tt_output_ttlg_dims, list_configurations[i].info_perms_output);
            // cost_trans_C = ttlg_transpose_time(len_output, list_configurations[i].info_dims_output, list_configurations[i].info_perms_output);
            // printf ("[%3d] >>> Output: %f\n", i, tmp);
          //  for (int j = 0; j < 6; j++)
          //      printf ("%d ", list_configurations[i].info_perms_output[j]);
            cost_transpositions += cost_trans_C;
            // cost_transpositions += 1000 * ttlg_transpose_time(len_output, tmp_tt_output_ttlg_dims, list_configurations[i].info_perms_output);
        }
       // printf ("\n");
        // printf ("# of Config. %d: %f\t %f\t %f: %f\n", i, cost_trans_A, cost_trans_B, cost_trans_C, cost_transpositions);

        //
        if (min_total_cost_AB > cost_transpositions)
        {
            min_total_cost_AB   = cost_transpositions;
            picked_cost_trans_A = cost_trans_A;
            picked_cost_trans_B = cost_trans_B;
            picked_cost_trans_C = cost_trans_C; 
            best_config_idx_AB = i;
        }
        // printf ("====================================================================================================================================\n");
    }
    // printf (">(AB)> best configuration: %d (idx #) among %d (its cost: %f (%f, %f, %f))\n", best_config_idx_AB, num_configurations, min_total_cost_AB, picked_cost_trans_A, picked_cost_trans_B, picked_cost_trans_C);

    //
    //  Type: BA
    //
    for (int i = num_configurations; i < num_configurations * 2; i++)
    {
        int     transpose_input_left    = OPT_TRANSPOSE_NO;
        int     transpose_input_right   = OPT_TRANSPOSE_NO;
        int     transpose_output        = OPT_TRANSPOSE_NO;
        double  cost_transpositions     = 0.0;
        double  cost_trans_A            = 0.0;
        double  cost_trans_B            = 0.0;
        double  cost_trans_C            = 0.0;

        //  INPUT-LEFT
        for (int j = 0; j < len_input_left; j++)
        {
            if (list_configurations[i].info_perms_input_right[j] != j)
            {
                transpose_input_left = OPT_TRANSPOSE_YES;
                break;
            }
        }

        //  INPUT-RIGHT
        for (int j = 0; j < len_input_right; j++)
        {
            if (list_configurations[i].info_perms_input_left[j] != j)
            {
                transpose_input_right = OPT_TRANSPOSE_YES;
                break;
            }
        }

        //  OUTPUT
        for (int j = 0; j < len_output; j++)
        {
            if (list_configurations[i].info_perms_output[j] != j)
            {
                transpose_output = OPT_TRANSPOSE_YES;
                break;
            }
        }

        // printf ("====================================================================================================================================\n");
        //  INPUT-LEFT
        if (transpose_input_right == OPT_TRANSPOSE_YES)
        {
            // cost_trans_A = ttlg_transpose_time(len_input_right, tmp_tt_input_right_ttlg_dims, list_configurations[i].info_perms_input_left);
         //   for (int j = 0; j < 4; j++)
          //      printf ("%d ", list_configurations[i].info_perms_input_left[j]);
            // printf ("[%3d] >>> Left: %f\n", i, tmp);
            cost_transpositions += cost_trans_A;
            // cost_transpositions += 1000 * ttlg_transpose_time(len_input_left, tmp_tt_input_left_ttlg_dims, list_configurations[i].info_perms_input_left);
        }
        
        //printf (" - ");

        //  INPUT-RIGHT
        if (transpose_input_left == OPT_TRANSPOSE_YES)
        {
            // cost_trans_B = ttlg_transpose_time(len_input_left, tmp_tt_input_left_ttlg_dims, list_configurations[i].info_perms_input_right);
            // printf ("[%3d] >>> Right: %f\n", i, tmp);
          //  for (int j = 0; j < 4; j++)
          //      printf ("%d ", list_configurations[i].info_perms_input_right[j]);
            cost_transpositions += cost_trans_B;
            // cost_transpositions += 1000 * ttlg_transpose_time(len_input_right, tmp_tt_input_right_ttlg_dims, list_configurations[i].info_perms_input_right);
        }

        //  DGEMM
 //       printf (" - ");

        //  OUTPUT
        if (transpose_output == OPT_TRANSPOSE_YES)
        {
            // cost_trans_C = 1000 * ttlg_transpose_time(len_output, tmp_tt_output_ttlg_dims, list_configurations[i].info_perms_output);
            // cost_trans_C = ttlg_transpose_time(len_output, list_configurations[i].info_dims_output, list_configurations[i].info_perms_output);
            // printf ("[%3d] >>> Output: %f\n", i, tmp);
           // for (int j = 0; j < 6; j++)
            //    printf ("%d ", list_configurations[i].info_perms_output[j]);
            cost_transpositions += cost_trans_C;
            // cost_transpositions += 1000 * ttlg_transpose_time(len_output, tmp_tt_output_ttlg_dims, list_configurations[i].info_perms_output);
        }
       // printf ("\n");
        
        // printf ("# of Config. %d: %f\t %f\t %f: %f\n", i, cost_trans_A, cost_trans_B, cost_trans_C, cost_transpositions);

        //
        if (min_total_cost_BA > cost_transpositions)
        {
            min_total_cost_BA   = cost_transpositions;
            picked_cost_trans_A = cost_trans_A;
            picked_cost_trans_B = cost_trans_B;
            picked_cost_trans_C = cost_trans_C; 
            best_config_idx_BA = i;
        }
        // printf ("====================================================================================================================================\n");
    }

    //
    // printf (">(BA)> best configuration: %d (idx #) among %d (its cost: %f (%f, %f, %f))\n", best_config_idx_BA, num_configurations, min_total_cost_BA, picked_cost_trans_A, picked_cost_trans_B, picked_cost_trans_C);

    //
    if (min_total_cost_AB >= min_total_cost_BA)
    {
        return best_config_idx_BA;
    }
    else
    {
        return best_config_idx_AB;
    }
}    


//
//  Model: Overall (Copy, TTLG and DGEMM)
//  To-Do: Need to be Modulized
//
int  tctt_model_overall(tc* info_tc, tiles*& result_tiles, tt*& result_tt_output, tt*& result_tt_input_left, tt*& result_tt_input_right, int target_configuration)
{
    //
    //  Permutations for Tile-Sizes
    //
    printf ("===========================================================================\n");

    //
    //  Given Information of Tensor Contraction.
    //
    int  len_output         = info_tc->len_output;
    int  len_input_left     = info_tc->len_input_left;
    int  len_input_right    = info_tc->len_input_right;
    idx* info_output        = info_tc->info_output;
    idx* info_input_left    = info_tc->info_input_left;
    idx* info_input_right   = info_tc->info_input_right;

    //
    //  Related to Memory
    //
    float size_gmem                 = (float)SIZE_GLOBAL_MEMORY;
    float size_tensor_output        = (float)((float)(info_tc->size_output)      / (float)(SIZE_MB)) * (float)(SIZE_TYPE);
    float size_tensor_input_left    = (float)((float)(info_tc->size_input_left)  / (float)(SIZE_MB)) * (float)(SIZE_TYPE);
    float size_tensor_input_right   = (float)((float)(info_tc->size_input_right) / (float)(SIZE_MB)) * (float)(SIZE_TYPE);
    float size_gmem_scratch         = size_gmem - (size_tensor_output + size_tensor_input_left + size_tensor_input_right);
    
#ifdef DEBUG_TILES
    printf ("GMEM: %f MB\n",            size_gmem);
    printf ("|Output|: %f MB\n",        size_tensor_output);
    printf ("|Input-Left|: %f MB\n",    size_tensor_input_left);
    printf ("|Input-Right|: %f MB\n",   size_tensor_input_right);
    printf ("All Tensors: %f MB\n",     size_tensor_output + size_tensor_input_left + size_tensor_input_right);
    printf ("Scratch: %f MB\n",         size_gmem_scratch);
#endif

    //
    idx* given_input_left   = info_tc->info_input_left;
    idx* given_input_right  = info_tc->info_input_right;

    // Need to Create Tile-Sizes which follows the constraints from the Full.
    int len_external_indices = info_tc->len_output;
    int len_internal_indices = ((info_tc->len_input_left + info_tc->len_input_right) - len_external_indices) / 2;

#ifdef DEBUG_TILES
    printf ("len. ext. idx.: %d\n", len_external_indices);
    printf ("len. int. idx.: %d\n", len_internal_indices);
#endif

    idx* temp_tiles_external = (idx*)malloc(sizeof(idx) * len_external_indices);
    idx* temp_tiles_intenral = (idx*)malloc(sizeof(idx) * len_internal_indices);

    for (int i = 0; i < len_external_indices; i++)
    {
        strncpy(temp_tiles_external[i].name, info_tc->info_output[i].name, SIZE_NAME);
        temp_tiles_external[i].size = info_tc->info_output[i].size;
    }

    for (int i = 0, int_i = 0; i < info_tc->len_input_left; i++)
    {
        if (find_index(info_tc->info_input_left[i].name, info_tc->info_output, info_tc->len_output) == -1)
        {
            strncpy(temp_tiles_intenral[int_i].name, info_tc->info_input_left[i].name, SIZE_NAME);
            temp_tiles_intenral[int_i++].size = info_tc->info_input_left[i].size;
        }
    }
#ifdef DEBUG_TILES_
    for (int i = 0; i < len_external_indices; i++)
        printf ("%s (%d), ", temp_tiles_external[i].name, temp_tiles_external[i].size);
    printf ("\n");
    for (int i = 0; i < len_internal_indices; i++)
        printf ("%s (%d), ", temp_tiles_intenral[i].name, temp_tiles_intenral[i].size);
    printf ("\n");
#endif

    //
    int check_sizes             = 1;
    int size_slice_output       = 1;
    int size_slice_input_left   = 1;
    int size_slice_input_right  = 1;
    int size_dgemm_m            = 1;
    int size_dgemm_n            = 1;
    int size_dgemm_k            = 1;

    //
    //  To Manipulate Tile-Sizes.
    //
    while (check_sizes > 0)
    {
        //  [Slice] Output
        for (int i = 0; i < len_output; i++)
        {
            if (find_size(info_output[i].name, temp_tiles_external, len_external_indices) != 0)
            {
                size_slice_output *= find_size(info_output[i].name, temp_tiles_external, len_external_indices);
            }
        }
        
        //  [Slice] Input-Left
        for (int i = 0; i < len_input_left; i++)
        {
            if (find_size(info_input_left[i].name, temp_tiles_external, len_external_indices) != 0)
            {
                int temp = find_size(info_input_left[i].name, temp_tiles_external, len_external_indices);
                size_slice_input_left   *= temp;
                size_dgemm_m            *= temp;
            }

            if (find_size(info_input_left[i].name, temp_tiles_intenral, len_internal_indices) != 0)
            {
                int temp = find_size(info_input_left[i].name, temp_tiles_intenral, len_internal_indices);
                size_slice_input_left   *= temp;
                size_dgemm_k            *= temp;
            }
        }

        //  [Slice] Input-Right
        for (int i = 0; i < len_input_right; i++)
        {
            if (find_size(info_input_right[i].name, temp_tiles_external, len_external_indices) != 0)
            {
                int temp = find_size(info_input_right[i].name, temp_tiles_external, len_external_indices);
                size_slice_input_right  *= temp;
                size_dgemm_n            *= temp;
            }

            if (find_size(info_input_right[i].name, temp_tiles_intenral, len_internal_indices) != 0)
            {
                size_slice_input_right  *= find_size(info_input_right[i].name, temp_tiles_intenral, len_internal_indices);
            }
        }

        //
        //printf ("===========================================================================\n");
        //printf (" |C'|: %f\n", (float)((float)(size_slice_output)        / (float)(SIZE_MB)) * (float)(SIZE_TYPE));
        //printf (" |A'|: %f\n", (float)((float)(size_slice_input_left)    / (float)(SIZE_MB)) * (float)(SIZE_TYPE));
        //printf (" |B'|: %f\n", (float)((float)(size_slice_input_right)   / (float)(SIZE_MB)) * (float)(SIZE_TYPE));
        printf ("[%d][GEMM] m: %d, n: %d, k: %d\n", check_sizes, size_dgemm_m, size_dgemm_n, size_dgemm_k);
        
        //
        //  Constraints
        //
        if (check_Constraints(size_slice_output, 1, size_slice_input_left, 1, size_slice_input_right, 1, size_gmem_scratch) != -1)
        {
            printf ("[%s][Constraints][PASSED] Tried %d times\n", __func__, check_sizes);
            check_sizes = -1;
        }
        else
        {
            printf ("[%s][Constraints][ERROR] Tried %d times\n", __func__, check_sizes);
            //
            //  To-Do: Actually, need to check other indices with the same size.
            //      Try to reduce the size of SVI. 
            //
            for (int i = (len_external_indices - 1); i >= 0; i--)
            {
                if (temp_tiles_external[i].size != 1)
                    temp_tiles_external[i].size = temp_tiles_external[i].size / 2;
                    break;
            }

            size_slice_output       = 1;
            size_slice_input_left   = 1;
            size_slice_input_right  = 1;
            size_dgemm_m            = 1;
            size_dgemm_n            = 1;
            size_dgemm_k            = 1;
            check_sizes++;
        }
        printf ("===========================================================================\n");
    }

    //
    //  result_tiles (This Process Might be Combined with the Below Process.)
    //
    result_tiles                            = (tiles*)malloc(sizeof(tiles));
    result_tiles->info_slice                = (idx*)malloc(sizeof(idx) * (len_external_indices + len_internal_indices));
    result_tiles->size_output_slice         = size_slice_output;
    result_tiles->size_input_left_slice     = size_slice_input_left;
    result_tiles->size_input_right_slice    = size_slice_input_right;
    result_tiles->dgemm_m                   = size_dgemm_m;
    result_tiles->dgemm_n                   = size_dgemm_n;
    result_tiles->dgemm_k                   = size_dgemm_k;

    printf ("result_tiles: m: %d, n: %d, k: %d\n", result_tiles->dgemm_m, result_tiles->dgemm_n, result_tiles->dgemm_k);
    printf ("===========================================================================\n");
    //
    //  info_slice
    //


    //
    //  Permutations for Tensor Transpositions.
    //
    printf ("[Picked Tile-Sizes] External: ");
    for (int i = 0; i < len_external_indices; i++)
    {
        printf ("%s(%d), ", temp_tiles_external[i].name, temp_tiles_external[i].size);
        strncpy(result_tiles->info_slice[i].name, temp_tiles_external[i].name, SIZE_NAME);
        result_tiles->info_slice[i].size = temp_tiles_external[i].size;
    }
    printf ("\n[Picked Tile-Sizes] Internal: ");
    for (int i = 0; i < len_internal_indices; i++)
    {
        printf ("%s(%d), ", temp_tiles_intenral[i].name, temp_tiles_intenral[i].size);
        strncpy(result_tiles->info_slice[i + len_external_indices].name, temp_tiles_intenral[i].name, SIZE_NAME);
        result_tiles->info_slice[i + len_external_indices].size = temp_tiles_intenral[i].size;
    }
    printf ("\n");

    //
    printf ("===========================================================================\n");
    //
    int min_overall_cost    = 1000000;
    int num_perms_tiles     = 1;
    int num_perms_trans     = 1;

    

    //  [1-1] Input-Left
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
    int*    idx_perms_int_input_left = (int*)malloc(sizeof(int) * num_perms_int_input_left * num_int_input_left);    /// Common for Both inputs

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

    //  [1-2] Input-Right
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
    //  [1-2-2] No Need to Make Perms. of Internal Indices in Input-Right,
    //          Because it should be identical to the one of internal indices in Input-Left.
    //
    offset = 0;
    /*
    perm(idx_int_input_right, num_int_input_right, 0, &offset, idx_perms_int_input_right);
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
*/
    //
    //
    //
    for (int i = 0; i < num_perms_int_input_left; i++)
    {
        for (int j = 0; j < num_int_input_left; j++)
        {
            for (int k = 0; k < num_ext_input_right + num_int_input_right; k++)
            {
                if (strcmp(given_input_right[k].name, given_input_left[idx_perms_int_input_left[i * num_int_input_left + j]].name) == 0)
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
    num_perms_trans = factorial(num_ext_input_left) * 2 * factorial(num_int_input_left) * factorial(num_ext_input_right) * 2;
#ifdef DEBUG_PERMUTATIONS
    printf ("There are %d different ways to transpose inputs.\n", num_perms_trans);
#endif

    //
    //  [1-3] Create a Target Output
    //
    int target_output[len_output][2];   // 0: Left, 1: Right
    int temp;
    for (int i = 0; i < len_output; i++)
    {
        temp = find_index(info_output[i].name, info_input_left, len_input_left);
        if (temp != -1)
        {
            target_output[i][0] = 0;        // Input-Left
            target_output[i][1] = temp;
        }

        temp = find_index(info_output[i].name, info_input_right, len_input_right);
        if (temp != -1)
        {
            target_output[i][0] = 1;        // Input-Right
            target_output[i][1] = temp;
        }
    }

#ifdef DEBUG_PERMUTATIONS
    printf ("Target Output: ");
    for (int i = 0; i < len_output; i++)
    {
        if (target_output[i][0] == 0)
        {
            printf ("[LEFT][%d], ", target_output[i][1]);
        }
        else
        {
            printf ("[Right][%d], ", target_output[i][1]);
        }
    }
    printf ("\n");
#endif

    //
    //  Input-Left: Default[0, 1, 2, 3] >>> [0, 1, 3, 2], [0, 2, 1, 3], [0, 2, 3, 1]
    //    
    //  idx_perms_ext_input_left, idx_perms_ext_input_right, idx_perms_intenral
    int perm_offset = 0;
    int result_output[len_output][2];
    int idx_perms_output[len_output];
    int idx_perms_input_left[len_input_left];
    int idx_perms_input_right[len_input_right];
    //
    info_config* list_configurations = (info_config*)malloc(sizeof(info_config) * num_perms_trans);
    
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
                        }
                        //  RIGHT
                        for (int i = 0; i < num_ext_input_right; i++)
                        {
                            result_output[i + num_ext_input_left][0] = 1;
                            result_output[i + num_ext_input_left][1] = idx_perms_ext_input_right[perm_ext_input_right * num_ext_input_right + i];
                        }
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
                        //  If 0,1,2,3,..,n, Then, Tensor-Transposition is not needed.
                        //
                        list_configurations[perm_offset].info_perms_output      = (int*)malloc(sizeof(int) * len_output);
                        list_configurations[perm_offset].info_perms_input_left  = (int*)malloc(sizeof(int) * len_input_left);
                        list_configurations[perm_offset].info_perms_input_right = (int*)malloc(sizeof(int) * len_input_right);

                        //  Copy The Output-Permutation
                        for (int i = 0; i < len_output; i++)
                        {
                            list_configurations[perm_offset].info_perms_output[i] = idx_perms_output[i];
                        }
                    
                    #ifdef DEBUG_PERMUTATIONS_DETAIL_INNER
                        printf ("[%4d] Output's Perm.: ", perm_offset);
                        for (int i = 0; i < len_output; i++)
                        {
                            printf ("%d [%d], ", idx_perms_output[i], list_configurations[perm_offset].info_perms_output[i]);
                        }
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
                                list_configurations[perm_offset].info_perms_input_left[i] = idx_perms_input_left[i];
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
                                    list_configurations[perm_offset].info_perms_input_right[i] = idx_perms_input_right[i];
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
                                    list_configurations[perm_offset].info_perms_input_right[i] = idx_perms_input_right[i];
                                }

                            #ifdef DEBUG_PERMUTATIONS_DETAIL_INNER
                                printf ("[#2] LEFT: ");
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
                                list_configurations[perm_offset].info_perms_input_left[i] = idx_perms_input_left[i];
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
                                    list_configurations[perm_offset].info_perms_input_right[i] = idx_perms_input_right[i];
                                }

                            #ifdef DEBUG_PERMUTATIONS_DETAIL_INNER
                                printf ("[#3] LEFT: ");
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
                                    list_configurations[perm_offset].info_perms_input_right[i] = idx_perms_input_right[i];
                                }

                            #ifdef DEBUG_PERMUTATIONS_DETAIL_INNER
                                printf ("[#4] LEFT: ");
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
    //  [Model]
    //
    printf ("#: %4d (%4d)\n", num_perms_trans, perm_offset);
    int best_configuration_offset   = 0;
    int min_total_cost              = 100000000;
    for (int i = 0; i < num_perms_trans; i++)
    {
        //
        //  [1-1] t2 -> t2': Cost-Extract
        //
        list_configurations[i].cost_cudaMemcpy_input_left   = tctt_model_extract_tensor();

        //
        //  [1-2] v2 -> v2': Cost-Extract
        //
        list_configurations[i].cost_cudaMemcpy_input_right  = tctt_model_extract_tensor();

        //
        //  [2-1] t2' -> t2'': Cost-Transpose
        //
        list_configurations[i].cost_transpose_input_left    = tctt_model_transpose_tensor();

        //
        //  [2-2] v2' -> v2'': Cost-Transpose
        //
        list_configurations[i].cost_transpose_input_right   = tctt_model_transpose_tensor();

        //
        //  [3-] DGEMM with (t2'|t2'') and (v2'|v2'')
        //  
        list_configurations[i].cost_dgemm                   = tctt_model_dgemm(16, 16, 0, 16, 16, 0);

        //
        //  [4-1] t3'' -> t3': Cost-Transpose
        //
        list_configurations[i].cost_transpose_output        = tctt_model_transpose_tensor();

        //
        //  [5-1] t3' -> t3: Cost-Extract
        //
        list_configurations[i].cost_cudaMemcpy_output       = tctt_model_extract_tensor();

        //
        list_configurations[i].total_cost = list_configurations[i].cost_cudaMemcpy_input_left + list_configurations[i].cost_cudaMemcpy_input_right + list_configurations[i].cost_cudaMemcpy_output +
                                            list_configurations[i].cost_transpose_input_left  + list_configurations[i].cost_transpose_input_right  + list_configurations[i].cost_transpose_output  +
                                            list_configurations[i].cost_dgemm; 
        
        //
        //
        //
        if (min_total_cost >= list_configurations[i].total_cost)
        {
            min_total_cost              = list_configurations[i].total_cost;
            best_configuration_offset   = i;
        }
        //printf ("[%4d] Total Cost: %d\n", i, list_configurations[i].total_cost);
    }    
    printf ("===========================================================================\n");

    //
    free(idx_perms_ext_input_left);
    free(idx_perms_int_input_left);
    free(idx_perms_ext_input_right);
    free(idx_perms_int_input_right);

    int picked_configuration = target_configuration;
    printf ("[Min. Cost] %8d by (%d)\n", min_total_cost, best_configuration_offset);
    printf ("[(Manually) Configuration # in %4d] %4d\n", num_perms_trans, picked_configuration);

//#ifdef DEBUG_PERMUTATIONS_RESULT
    // Output
    printf ("[Output] ");
    for (int i = 0; i < len_output; i++)
    {
        printf ("%d, ", list_configurations[picked_configuration].info_perms_output[i]);
    }
    printf ("\n");

    // Input-Left
    printf ("[Input][LEFT] ");
    for (int i = 0; i < len_input_left; i++)
    {
        printf ("%d, ", list_configurations[picked_configuration].info_perms_input_left[i]);
    }
    printf ("\n");

    // Input-Right
    printf ("[Input][RIGHT] ");
    for (int i = 0; i < len_input_right; i++)
    {
        printf ("%d, ", list_configurations[picked_configuration].info_perms_input_right[i]);
    }
    printf ("\n");
//#endif
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
        if (list_configurations[picked_configuration].info_perms_output[i] != i)
        {
            printf ("[Output] Transposition is Required\n");
            result_tt_output->transpose         = OPT_TRANSPOSE_YES;
            break;
        }
    }

    //
    //  Input-Tensor (LEFT)
    //
    for (int i = 0; i < len_input_left; i++)
    {
        if (list_configurations[picked_configuration].info_perms_input_left[i] != i)
        {
            printf ("[Input][Left] Transposition is Required\n");
            result_tt_input_left->transpose         = OPT_TRANSPOSE_YES;
            break;
        }
    }
    
    //
    if (find_size(info_input_left[list_configurations[picked_configuration].info_perms_input_left[0]].name, temp_tiles_intenral, len_internal_indices) == 0)
    {
        //  The FVI of t2' is not an Internal Index, resulting in (E_A, K) Form (N)
        result_tt_input_left->dgemm_transpose   = OPT_TRANSPOSE_NO;     // (default) N
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
        if (list_configurations[picked_configuration].info_perms_input_right[i] != i)
        {
            printf ("[Input][Right] Transposition is Required\n");
            result_tt_input_right->transpose = OPT_TRANSPOSE_YES;
            break;
        }
    }
    
    //
    if (find_size(info_input_right[list_configurations[picked_configuration].info_perms_input_right[0]].name, temp_tiles_intenral, len_internal_indices) == 0)
    {
        //  The FVI of v2' is not an Internal Index, resulting in (E_B, K) Form (T)
        result_tt_input_right->dgemm_transpose  = OPT_TRANSPOSE_YES;    // (default) T
    }
    else
    {
        //  The FVI of v2' is an Internal Index, resulting in (K, E_B) Form (N)
        result_tt_input_right->dgemm_transpose  = OPT_TRANSPOSE_NO;
    }


    //
    result_tt_output->len_indice        = len_output;
    result_tt_input_left->len_indice    = len_input_left;
    result_tt_input_right->len_indice   = len_input_right;

    result_tt_output->ttlg_dims         = (int*)malloc(sizeof(int) * len_output);
    result_tt_output->ttlg_perms        = (int*)malloc(sizeof(int) * len_output);

    result_tt_input_left->ttlg_dims     = (int*)malloc(sizeof(int) * len_input_left);
    result_tt_input_left->ttlg_perms    = (int*)malloc(sizeof(int) * len_input_left);

    result_tt_input_right->ttlg_dims    = (int*)malloc(sizeof(int) * len_input_right);
    result_tt_input_right->ttlg_perms   = (int*)malloc(sizeof(int) * len_input_right);

    //idx* info_output        = info_tc->info_output;
    //idx* info_input_left    = info_tc->info_input_left;
    //idx* info_input_right   = info_tc->info_input_right;
    //int temp = find_size(info_input_left[i].name, temp_tiles_external, len_external_indices);


    //
    //  [To-Do] Should be Fixed 
    //
    for (int i = 0; i < len_output; i++)
    {
        result_tt_output->ttlg_perms[i]         = list_configurations[picked_configuration].info_perms_output[i];
        result_tt_output->ttlg_dims[list_configurations[picked_configuration].info_perms_output[i]] = find_size(info_output[i].name, temp_tiles_external, len_external_indices);
    }

    //
    for (int i = 0; i < len_input_left; i++)
    {
        int tmp_dim = find_size(info_input_left[i].name, temp_tiles_external, len_external_indices);
        if (tmp_dim > 0)
        {
            result_tt_input_left->ttlg_dims[i]      = tmp_dim;
        }
        else
        {
            result_tt_input_left->ttlg_dims[i]      = find_size(info_input_left[i].name, temp_tiles_intenral, len_internal_indices);
        }
        
        result_tt_input_left->ttlg_perms[i]     = list_configurations[picked_configuration].info_perms_input_left[i];
    #ifdef DEBUG_MAKE_INFO_TTLG
        printf ("info_input_left[%d][%s][%d] vs result_tt_input_left[%d][%d]\n", i, info_input_left[i].name, info_input_left[i].size, i, result_tt_input_left->ttlg_dims[i]);   
    #endif
    }

    //
    for (int i = 0; i < len_input_right; i++)
    {
        int tmp_dim = find_size(info_input_right[i].name, temp_tiles_external, len_external_indices);
        if (tmp_dim > 0)
        {
            result_tt_input_right->ttlg_dims[i]     = tmp_dim;
        }
        else
        {
            result_tt_input_right->ttlg_dims[i]     = find_size(info_input_right[i].name, temp_tiles_intenral, len_internal_indices);
        }
        
        result_tt_input_right->ttlg_perms[i]    = list_configurations[picked_configuration].info_perms_input_right[i];
    #ifdef DEBUG_MAKE_INFO_TTLG
        printf ("info_input_right[%d][%s][%d] vs result_tt_input_right[%d][%d]\n", i, info_input_right[i].name, info_input_right[i].size, i, result_tt_input_right->ttlg_dims[i]);
    #endif
    }

    //
    return min_total_cost;
}

//
int tctt_model_extract_tensor()
{

    return 2;
}

//
int tctt_model_transpose_tensor()
{
    
    return 1;
}

//
//  A: m * k
//  B: k * n
//  # of Operations: m * k * n * 2
//
int tctt_model_dgemm(int row_A, int column_A, int trans_A, int row_B, int column_B, int trans_B)
{
    unsigned int num_operations = 0;

    //
    //  trans_A: 0 (N), 1 (T)
    //  trans_B: 0 (N), 1 (N)
    //

    //  A: N
    if (trans_A == 0)
    {
        //  B: N
        if (trans_B == 0)
        {
            //
            //  A (N) && B(N)
            //

        }
        //  B: T
        else
        {
            //
            //  A(N) && B(T)
            //

        }
    }
    //  A: T
    else
    {
        //  B: N
        if (trans_B == 0)
        {
            //
            //  A(T) && B (N)
            //

        }
        //  B: T
        else
        {
            //
            //  A(T) && B(T)
            //

        }
    }

    return 3;
}
