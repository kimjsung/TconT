#include "../ttgt_ttlg.h"

//#define DEBUG_BASE
#define DEBUG_ENUM
//#define DEBUG_ENUM_DETAIL

//
//  To figure out the best tiling fitted in HW_SCRATCH
//  Assumption: It is called after checking that all configuration based on TTGT cannot be feasible given the scratch memory.
//
void tiling_scratch(tc* info_tc)
{
    // 
    //  to extract info. from "(tc*) info_tc"
    //
    int  len_output         = info_tc->len_output;
    int  len_input_left     = info_tc->len_input_left;
    int  len_input_right    = info_tc->len_input_right;
    idx* info_output        = info_tc->info_output;
    idx* info_input_left    = info_tc->info_input_left;
    idx* info_input_right   = info_tc->info_input_right;

    //
    int len_external_indices = len_output;
    int len_internal_indices = ((len_input_left + len_input_right) - len_external_indices) / 2;

    //  Given volumes of A, B, and C,
    double vol_C = ((double)(info_tc->size_output)      / (double)(SIZE_MB)) * SIZE_TYPE;
    double vol_A = ((double)(info_tc->size_input_left)  / (double)(SIZE_MB)) * SIZE_TYPE;
    double vol_B = ((double)(info_tc->size_input_right) / (double)(SIZE_MB)) * SIZE_TYPE;

    double vol_total    = vol_C + vol_A + vol_B;
    double vol_scratch  = (vol_total * HW_PERCENTAGE) / 100;

#ifdef DEBUG_BASE
    printf ("===========================================================================\n");
    printf ("|A| + |B| + |C| = %.4f + %.4f + %.4f = %.4f MB\n", vol_A, vol_B, vol_C, vol_total);
    printf ("Scracth: %3d\% = %.4f MB\n", HW_PERCENTAGE, vol_scratch);
    printf ("===========================================================================\n");
#endif

    /*
     *
        Tiling is based on only external indices.
     *
     */
    idx* info_tiling = (idx*)malloc(sizeof(idx) * (len_external_indices + len_internal_indices));
    for (int i = 0; i < len_external_indices; i++)
    {
        strncpy(info_tiling[i].name, info_output[i].name, SIZE_NAME);
        //info_tiling[i].size         = info_output[i].size;
        info_tiling[i].size         = 32;
        info_tiling[i].tile_size    = info_output[i].tile_size;
        info_tiling[i].idx_type     = TYPE_EXTERNAL;
    }

    int offset_internal = 0;
    for (int i = 0; i < len_input_left; i++)
    {
        for (int j = 0; j < len_input_right; j++)
        {
            if (strcmp(info_input_left[i].name, info_input_right[j].name) == 0)
            {
                strncpy(info_tiling[len_external_indices + offset_internal].name, info_input_left[i].name, SIZE_NAME);
                //info_tiling[len_external_indices + offset_internal].size        = info_input_left[i].size;
                info_tiling[len_external_indices + offset_internal].size        = 32;
                info_tiling[len_external_indices + offset_internal].tile_size   = info_input_left[i].tile_size;
                info_tiling[len_external_indices + offset_internal].idx_type    = TYPE_INTERNAL;
                offset_internal++;
            }
        }
    }

#ifdef DEBUG_ENUM_DETAIL
    for (int i = 0; i < len_external_indices + len_internal_indices; i++)
    {
        if (info_tiling[i].idx_type == TYPE_INTERNAL)
            printf ("int. :");
        else
            printf ("ext. :");
        printf ("%s (%d) (%d)\n", info_tiling[i].name, info_tiling[i].size, info_tiling[i].tile_size);
    }
#endif
    int tmp_tiles[len_external_indices + len_internal_indices];
    //for (int i = 0; i < len_external_indices + len_internal_indices; i++)
    //    tmp_tiles[i] = 3;

    //
    //  Overall: It enumerates all possible tile-sizes.
    //  (1) Prune by Scratch
    //  (2) Prune by Constraints
    //  (2) Pick the best one by Model(s)
    //
    helper_recursive_tiling_idx(info_tc, 0, len_external_indices + len_internal_indices, info_tiling, tmp_tiles);
}

//
//  (1) To enumerate all possible tile sizes by searching recursively.
//
void helper_recursive_tiling_idx(tc* info_tc, int idx_offset, int idx_bound, idx* info_tiling, int* tiles)
{
    if (idx_offset < idx_bound)
    {
        //
        //  (temporally) 1 <= T_idx <= S_idx // FMA (float) == diff. tile sizes
        //
        //for (int i = 0; i < info_tiling[idx_offset].size; i++)
        for (int i = 28; i <= info_tiling[idx_offset].size; i+=2)
        {   
            tiles[idx_offset] = i;
            helper_recursive_tiling_idx(info_tc, idx_offset + 1, idx_bound, info_tiling, tiles);
        }
    }
    else
    {
    #ifdef DEBUG_ENUM_DETAIL
        printf (">>>  ");
        for (int i = 0; i < idx_bound; i++)
            printf ("%s (%d), ", info_tiling[i].name, tiles[i]);
        printf ("\n");
    #endif
        
        //
        //  to check if the tiling info. can be fitted in the given scratch memory or not.
        //  it should return "tiling info" and/or "diff. enumeration for lottgt."
        //
        lottgt_enumeration(info_tc, info_tiling, tiles, idx_bound);

        //
        //  if the tiling info. is passed, then the tiling info. should be calculated by our model.
        //
    }
}

//
//  to check if the tiling info. can be gitted in the given scratch memory or not.
//
int helper_tiling_fit_scratch(tc* info_tc, idx* info_tiling, int* tiles, int idx_bound)
{
    double cal_size_C = 1.0;
    double cal_size_A = 1.0;
    double cal_size_B = 1.0;

    //
    //  C
    //
    for (int i = 0; i < info_tc->len_output; i++)
    {
        for (int j = 0; j < info_tc->len_output; j++)
        {
            if (strcmp(info_tc->info_output[i].name, info_tiling[j].name) == 0)
            {
            #ifdef DEBUG_ENUM_DETAIL
                printf ("C: %s (%d)\n", info_tiling[j].name, tiles[j]);
            #endif
                cal_size_C *= tiles[j];
            }
        }
    }
    cal_size_C = (cal_size_C * SIZE_TYPE) / (SIZE_MB);

    //
    //  A
    //
    for (int i = 0; i < info_tc->len_input_left; i++)
    {
        int tiling_idx = -1;
        for (int j = 0; j < info_tc->len_output; j++)
        {
            if (strcmp(info_tc->info_input_left[i].name, info_tiling[j].name) == 0)
            {
                tiling_idx = j;
                break;
            }
        }

        if (tiling_idx > -1)
        {
            //cal_size_A *= info_tiling[tiling_idx].tile_size;
        #ifdef DEBUG_ENUM_DETAIL
            printf ("A(ext.): %s (%d)\n", info_tiling[tiling_idx].name, tiles[tiling_idx]);
        #endif
            cal_size_A *= tiles[tiling_idx];
        }
        else
        {
        #ifdef DEBUG_ENUM_DETAIL
            printf ("A(int.): %s (%d)\n", info_tc->info_input_left[i].name, info_tc->info_input_left[i].size);
        #endif
            cal_size_A *= info_tc->info_input_left[i].size;
        }
    }
    cal_size_A = (cal_size_A * SIZE_TYPE) / (SIZE_MB);

    //
    //  B
    //
    for (int i = 0; i < info_tc->len_input_right; i++)
    {
        int tiling_idx = -1;
        for (int j = 0; j < info_tc->len_output; j++)
        {
            if (strcmp(info_tc->info_input_right[i].name, info_tiling[j].name) == 0)
            {
                tiling_idx = j;
                break;
            }
        }

        if (tiling_idx > -1)
        {
            //cal_size_B *= info_tiling[tiling_idx].tile_size;
        #ifdef DEBUG_ENUM_DETAIL
            printf ("B(ext.): %s (%d)\n", info_tiling[tiling_idx].name, tiles[tiling_idx]);
        #endif
            cal_size_B *= tiles[tiling_idx];
        }
        else
        {
        #ifdef DEBUG_ENUM_DETAIL
            printf ("B(int.): %s (%d)\n", info_tc->info_input_right[i].name, info_tc->info_input_right[i].size);
        #endif
            cal_size_B *= info_tc->info_input_right[i].size;
        }
    }
    cal_size_B = (cal_size_B * SIZE_TYPE) / (SIZE_MB);
    
    //
    //  Configurations might be different between TTGT and LoTTGT, according to tiling.
    // 
    lottgt_enumeration(info_tc, info_tiling, tiles, idx_bound);

    //
#ifdef DEBUG_ENUM
    printf ("According to the given tiling, |C| = %.4f, |A| = %.4f, |B| = %.4f >> |TOTAL| = %.4f > |Scratch| = %d\n", cal_size_C, cal_size_A, cal_size_B, (cal_size_C + cal_size_A + cal_size_B), HW_SCRATCH);
#endif
    if ((cal_size_C + cal_size_A + cal_size_B) <= HW_SCRATCH)
    {
        printf ("Tiling is feasible in the given scratch memory.\n");
        return 1; //  feasible
    }
    else
    {
        printf ("Tiling is NOT feasible in the given scratch memory.\n");
        return 0; //  infeasible
    }    
}

//
//  a loop-index's location is important.
//  multiple loop-indices can be possible.
//  
//  # 13: ab-acd-dbc (312,296,296,312): |A| = 219.83, |B| = 208.56,  |C| = 0.70, |TOTAL| = 429.10.
//
//      50% scratch: 214.55,
//      40% scratch: 171.64,
//      30% scratch: 128.73,
//      20% scratch:  85.82,
//      10% scratch:  42.91
//
void lottgt_enumeration(tc* info_tc, idx* info_tiling, int* tiles, int idx_bound)
{
    //
    //  T_i = 1 >>> Serialized
    //
    int diff_approach = TYPE_SAME_ENUM;
    for (int i = 0; i < idx_bound; i++)
    {
        if (tiles[i] == 1)
        {
            printf ("%2d index's tile size = %2d\n", i, tiles[i]);
            diff_approach = TYPE_DIFF_ENUM;
        }
    }

    //
    //
    //
    if (diff_approach == TYPE_DIFF_ENUM)
    {
        printf ("DIFF ENUMERATION TO TTGT\n");
        //
        //  to figure out different enumerations for LoTTGT
        //
    }
    else
    {
        printf ("SAME ENUMERATION TO TTGT\n");
        //
        //  Model for LoTTGT 
        //
    }
}