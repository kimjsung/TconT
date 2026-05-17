#include "../ttgt_ttlg.h"

//#define DEBUG_PRINT

//
//  A Tensor Contraction:
//  : C = alpha * A * B + beta * C;
//  (1) beta == 0.0
//      : C is not given,
//  (2) otherwise
//      : C is given, C should be accumulated.
//
//  When LoTTGT is required,
//  "Combination" --- nCr
//  C(3,1) = 3 -> (1), (2), and (3)
//  C(3,2) = 3 -> (4), (5), and (6)
//  C(3,3) = 1 -> (7)
//  These are all exclusive.
//
//  (1) A -> A'
//      (1-1) |A| > S
//      >> tiling external indices in A, 
//      >> tiling internal indices in A,
//      >> tiling both external and internal indices in A,
//  (2) B -> B'
//      (2-1) |B| > S
//  (3) C -> C'
//      (3-1) |C| > S
//  (4) A -> A' && B -> B'
//      (4-1) When the given A and B should be kept, |A| + |B| > S.
//      (4-2) The given A and B can be replaced by A' and B', max(|A|, |B|) > S.
//  (5) A -> A' && C' -> C
//      (5-1) max(|A|, |C|) > S.
//  (6) B -> B' && C' -> C
//      (6-1) max(|B|, |C|) > S.
//  (7) A -> A' && B -> B' && C' -> C
//      (7-1) When the given A and B should be kept, max(|A| + |B|, |C|) > S.
//      (7-2) The given A and B can be replaced by A' and B', max(|A|, |B|, |C|) > S.
//
int helper_ttgt_vs_lottgt(tc* info_tc, tt* picked_config_output, tt* picked_config_input_left, tt* picked_config_input_right, int opt_keep_tensors)
{
#ifdef DEBUG_PRINT
    printf ("===========================================================================\n");
    if (opt_keep_tensors == GIVEN_TENSORS_YES)
        printf ("[TTGT][LoTTGT] Given tensors should be kept.\n");
    else
        printf ("[TTGT][LoTTGT] Given tensors should be replaced.\n");
#endif

    int transpose_input_left    = picked_config_input_left->transpose;
    int transpose_input_right   = picked_config_input_right->transpose;
    int transpose_output        = picked_config_output->transpose;

    //
    //  |C|, |A|, |B| (based on MiB)
    //
    double size_C = (double)((double)(info_tc->size_output      * SIZE_TYPE) / (double)(SIZE_MB));
    double size_A = (double)((double)(info_tc->size_input_left  * SIZE_TYPE) / (double)(SIZE_MB));
    double size_B = (double)((double)(info_tc->size_input_right * SIZE_TYPE) / (double)(SIZE_MB));

#ifdef DEBUG_BASE
    printf ("|A|: %.4f\n", size_A);
    printf ("|B|: %.4f\n", size_B);
    printf ("|C|: %.4f\n", size_C);
    printf ("|S|: %d\n", HW_SCRATCH);
#endif

    //  (1) A -> A'
    if (transpose_input_left    == OPT_TRANSPOSE_YES    && 
        transpose_input_right   == OPT_TRANSPOSE_NO     && 
        transpose_output        == OPT_TRANSPOSE_NO)
    {
    #ifdef DEBUG_PRINT
        printf ("[1] A -> A': A, A', B or A', B will be located before GEMM\n");
        printf ("[Requirement #1] |A| < S\n");
        printf ("|A| > |S|: |A| = %.4f ?> %d\n", size_A, HW_SCRATCH);
    #endif
        if (size_A > HW_SCRATCH)
            return SCRATCH_UNFIT;
        else
            return SCRATCH_FIT;
    }

    //  (2) B -> B'
    if (transpose_input_left    == OPT_TRANSPOSE_NO     && 
        transpose_input_right   == OPT_TRANSPOSE_YES    && 
        transpose_output        == OPT_TRANSPOSE_NO)
    {
    #ifdef DEBUG_PRINT
        printf ("[2] B -> B': A, B, B' or A, B' will be located before GEMM\n");
        printf ("[Requirement #1] |B| < S\n");
        printf ("B| > |S|: |B| = %.4f ?> %d\n", size_B, HW_SCRATCH);
    #endif


        if (size_B > HW_SCRATCH)
            return SCRATCH_UNFIT;
        else
            return SCRATCH_FIT;
    }

    //  (3) C' -> C
    if (transpose_input_left    == OPT_TRANSPOSE_NO     && 
        transpose_input_right   == OPT_TRANSPOSE_NO     && 
        transpose_output        == OPT_TRANSPOSE_YES)
    {
    #ifdef DEBUG_PRINT
        printf ("[3] C' -> C: After GEMM, C', C will be located.\n");
        printf ("[Requirement #1] |C| < S\n");
        printf ("|C| > |S|: |C| = %.4f ?> %d\n", size_C, HW_SCRATCH);
    #endif
        if (size_C > HW_SCRATCH)
            return SCRATCH_UNFIT;
        else
            return SCRATCH_FIT;
    }

    //  (4) A -> A' && B -> B'
    if (transpose_input_left    == OPT_TRANSPOSE_YES    && 
        transpose_input_right   == OPT_TRANSPOSE_YES    && 
        transpose_output        == OPT_TRANSPOSE_NO)
    {
    #ifdef DEBUG_PRINT
        printf ("[4] A -> A' && B -> B': A, A', B, B' will be located before GEMM\n");
    #endif
        if (opt_keep_tensors == GIVEN_TENSORS_YES)
        {
        #ifdef DEBUG_PRINT
            printf ("[4-1] Both A and B should be kept\n");
            printf ("[4-1-1] (1) A -> A', (2) B -> B'\n");
            printf ("[Requirement #1] |A| < S\n");
            printf ("[Requirement #2] |B| < S - |A| = |A| + |B| < S\n");
            printf ("[4-1-2] (1) B -> B', (2) A -> A'\n");
            printf ("[Requirement #1] |B| < S\n");
            printf ("[Requirement #2] |A| < S - |B| = |A| + |B| < S\n");
            printf ("|A| + |B| = %.4f ?> %d\n", size_A + size_B, HW_SCRATCH);
        #endif
            if (size_A + size_B > HW_SCRATCH)
                return SCRATCH_UNFIT;
            else
                return SCRATCH_FIT;
        }
        else
        {
        #ifdef DEBUG_PRINT
            printf ("[4-2] Both A and B should be replaced by A' and B': A', B' will be located before GEMM\n");
            printf ("[4-2-1] (1) A -> A', (2) B -> B'\n");
            printf ("[Requirement #1] |A| > S\n");
            printf ("[Requirement #2] |B| > S\n");
            printf ("[4-2-2] (1) B -> B', (2) A -> A'\n");
            printf ("[Requirement #1] |B| > S\n");
            printf ("[Requirement #2] |A| > S\n");
            printf ("|A| > |S| or |B| > |S|: |A| = %.4f, |B| = %.4f ?> %d\n", size_A, size_B, HW_SCRATCH);
        #endif
            if (size_A > HW_SCRATCH || size_B > HW_SCRATCH)
                return SCRATCH_UNFIT;
            else
                return SCRATCH_FIT;
        }
    }

    //  (5)     A -> A' && C' -> C
    if (transpose_input_left    == OPT_TRANSPOSE_YES    && 
        transpose_input_right   == OPT_TRANSPOSE_NO     && 
        transpose_output        == OPT_TRANSPOSE_YES)
    {
    #ifdef DEBUG_PRINT
        printf ("[5] A -> A' && C' -> C\n");
    #endif    
        if (opt_keep_tensors == GIVEN_TENSORS_YES)
        {
        #ifdef DEBUG_PRINT
            printf ("[5-1] A should be kept\n");
            printf ("[Requirement #1] |A| < S\n");
            printf ("[Requirement #2] |C| < S - |A| = |A| + |C| < S\n");
            printf ("|A| + |C| = %.4f ?> %d\n", size_A + size_C, HW_SCRATCH);
        #endif
            if (size_A + size_C > HW_SCRATCH)
                return SCRATCH_UNFIT;
            else
                return SCRATCH_FIT;
        }
        else
        {
            #ifdef DEBUG_PRINT
            printf ("[5-2] A should be replaced by A'\n");
            printf ("[Requirement #1] |A| > S\n");
            printf ("[Requirement #2] |C| > S\n");
            printf ("|A| > |S| or |C| > |S|: |A| = %.4f, |C| = %.4f ?> %d\n", size_A, size_C, HW_SCRATCH);
        #endif
            if (size_A > HW_SCRATCH || size_C > HW_SCRATCH)
                return SCRATCH_UNFIT;
            else
                return SCRATCH_FIT;
        }
    }

    //  (6) B -> B' && C' -> C
    if (transpose_input_left    == OPT_TRANSPOSE_NO     && 
        transpose_input_right   == OPT_TRANSPOSE_YES    && 
        transpose_output        == OPT_TRANSPOSE_YES)
    {
    #ifdef DEBUG_PRINT
        printf ("[6] B -> B' && C' -> C\n");
    #endif
        if (opt_keep_tensors == GIVEN_TENSORS_YES)
        {
        #ifdef DEBUG_PRINT
            printf ("[6-1] B should be kept\n");
            printf ("[Requirement #1] |B| < S\n");
            printf ("[Requirement #2] |C| < S - |B| = |B| + |C| < S\n");
            printf ("|B| + |C| = %.4f ?> %d\n", size_B + size_C, HW_SCRATCH);
        #endif
            if (size_B + size_C > HW_SCRATCH)
                return SCRATCH_UNFIT;
            else
                return SCRATCH_FIT;
        }
        else
        {
        #ifdef DEBUG_PRINT
            printf ("[6-2] B should be replaced by B'\n");
            printf ("[Requirement #1] |B| > S\n");
            printf ("[Requirement #2] |C| > S\n");
            printf ("|B| > |S| or |C| > |S|: |B| = %.4f, |C| = %.4f ?> %d\n", size_B, size_C, HW_SCRATCH);
        #endif
            if (size_B > HW_SCRATCH || size_C > HW_SCRATCH)
                return SCRATCH_UNFIT;
            else
                return SCRATCH_FIT;
        }
    }

    //  (7) A -> A' && B -> B' && C' -> C
    if (transpose_input_left    == OPT_TRANSPOSE_YES    && 
        transpose_input_right   == OPT_TRANSPOSE_YES    && 
        transpose_output        == OPT_TRANSPOSE_YES)
    {
    #ifdef DEBUG_PRINT
        printf ("[7] A -> A' && B -> B' && C' -> C\n");
    #endif
        if (opt_keep_tensors == GIVEN_TENSORS_YES)
        {
        #ifdef DEBUG_PRINT
            printf ("[7-1] Both A and B should be kept\n");
            printf ("[7-1-1] (1) A -> A', (2) B -> B'\n");
            printf ("[Requirement #1] |A| < S\n");
            printf ("[Requirement #2] S - |A| < |B| = |A| + |B| < S\n");
            printf ("[Requirement #3] S - |A| - |B| > |C| = |A| + |B| + |C| < S\n");
            printf ("[7-1-2] (1) B -> B', (2) A -> A'\n");
            printf ("[Requirement #1] |B| < S\n");
            printf ("[Requirement #2] S - |B| < |A| = |A| + |B| < S\n");
            printf ("[Requirement #3] S - |B| - |A| > |C| = |B| + |A| + |C| < S\n");
            printf ("|A| + |B| + |C| = %.4f ?> %d\n", size_A + size_B + size_C, HW_SCRATCH);
        #endif
            if (size_A + size_B + size_C > HW_SCRATCH)
                return SCRATCH_UNFIT;
            else
                return SCRATCH_FIT;
        }
        else
        {
        #ifdef DEBUG_PRINT
            printf ("[7-2] Both A and B should be replaced by A' and B'\n");
            printf ("[7-2-1] (1) A -> A', (2) B -> B'\n");
            printf ("[Requirement #1] |A| > S\n");
            printf ("[Requirement #2] |B| > S\n");
            printf ("[Requirement #3] |C| > S\n");
            printf ("[7-2-2] (1) B -> B', (2) A -> A'\n");
            printf ("[Requirement #1] |A| > S\n");
            printf ("[Requirement #2] |B| > S\n");
            printf ("[Requirement #3] |C| > S\n");
            printf ("|A| > |S| or |B| > |S| or |C| > |S|: |A| = %.4f, |B| = %.4f, |C| = %.4f ?> %d\n", size_A, size_B, size_C, HW_SCRATCH);
        #endif
            if (size_A > HW_SCRATCH || size_B > HW_SCRATCH || size_C > HW_SCRATCH)
                return SCRATCH_UNFIT;
            else
                return SCRATCH_FIT;
        }
    }
    printf ("===========================================================================\n");
    return SCRATCH_UNFIT; //  If it is not matched, it is not feasible.
}   

