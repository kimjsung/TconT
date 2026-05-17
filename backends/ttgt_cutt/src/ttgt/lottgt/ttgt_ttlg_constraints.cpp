#include "../ttgt_ttlg.h"

#define DEBUG_CONSTRAINTS

// 
int  check_Constraints(int size_c_slice, int is_t_c, int size_a_slice, int is_t_a, int size_b_slice, int is_t_b, float mem_size_scratch)
{
    float mem_size_c_slice = ((float)(size_c_slice) / (float)(SIZE_MB))  * SIZE_TYPE;
    float mem_size_a_slice = (float)(size_a_slice * SIZE_TYPE) / (float)(SIZE_MB);
    float mem_size_b_slice = (float)(size_b_slice * SIZE_TYPE) / (float)(SIZE_MB);

#ifdef DEBUG_CONSTRAINTS
    printf ("===========================================================================\n");
    printf (">>> Slices\n");
    printf (">>> |C'| = %'15d (# of Elements) >>> %'12.4f MB\n", size_c_slice, mem_size_c_slice);
    printf (">>> |A'| = %'15d (# of Elements) >>> %'12.4f MB\n", size_a_slice, mem_size_a_slice);
    printf (">>> |B'| = %'15d (# of Elements) >>> %'12.4f MB\n", size_b_slice, mem_size_b_slice);
    printf (">>> |Total|                               >>> %'12.4f MB\n", mem_size_c_slice + mem_size_a_slice + mem_size_b_slice);
    printf (">>> |S|                                   >>> %'12.4f MB\n", mem_size_scratch);
    printf ("===========================================================================\n");
#endif

    // (1) C'' + B'' + A'' <= S, where S is the size of scratch.
    if ((mem_size_c_slice + mem_size_a_slice + mem_size_b_slice) > mem_size_scratch)
    {
#ifdef DEBUG_CONSTRAINTS
        printf ("(Q)|A'| + |B'| + |C'| <= |S|?, %'12.4f MB > %'12.4f MB\n", mem_size_c_slice + mem_size_a_slice + mem_size_b_slice, mem_size_scratch);
        printf ("[Constractins][1] ERROR!\n");
#endif 
        return -1;
    }
#ifdef DEBUG_CONSTRAINTS
    else
    {
        printf ("(Q) |A'| + |B'| + |C'| <= |S|?, %'12.4f MB <= %'12.4f MB\n", mem_size_a_slice + mem_size_b_slice + mem_size_c_slice, mem_size_scratch);
        printf ("[Constraints][1] PASSED\n");
    }
#endif
    // (2) A' + A'' <= S if A' != A''.
    if (is_t_a == 1)
    {
        if (mem_size_a_slice * 2 > mem_size_scratch)
        {
#ifdef DEBUG_CONSTRAINTS
            printf ("(Q) |A'| + |A''| <= |S|?, %'12.4f MB > %'12.4f MB\n", mem_size_a_slice * 2, mem_size_scratch);
            printf ("[Constraints][2] ERROR!\n");
#endif
            return -1;
        }
#ifdef DEBUG_CONSTRAINTS        
        else
        {
            printf ("(Q) |A'| + |A''| <= |S|?, %'12.4f MB <= %'12.4f MB\n", mem_size_a_slice * 2, mem_size_scratch);
            printf ("[Constraints][2] PASSED!\n");
        }
#endif
    }

    // (3) B' + B'' <= S if B' != B''.
    if (is_t_b == 1)
    {
        if (mem_size_b_slice * 2 > mem_size_scratch)
        {
#ifdef DEBUG_CONSTRAINTS
            printf ("(Q) |B'| + |B''| <= |S|?, %'12.4f MB > %'12.4f MB\n", mem_size_b_slice * 2, mem_size_scratch);
            printf ("[Constraints][3] ERROR!\n");
#endif
            return -1;
        }
#ifdef DEBUG_CONSTRAINTS
        else
        {
            printf ("(Q) |B'| + |B''| <= |S|?, %'12.4f MB <= %'12.4f MB\n", mem_size_b_slice * 2, mem_size_scratch);
            printf ("[Constraints][3] PASSED!\n");
        }
#endif
    }

    // (4) C' + C'' <= S if C' != C''.
    if (is_t_c == 1)
    {
        if (mem_size_c_slice * 2 > mem_size_scratch)
        {
#ifdef DEBUG_CONSTRAINTS
            printf ("(Q) |C'| + |C''| <= |S|?, %'12.4f MB > %'12.4f MB\n", mem_size_c_slice * 2, mem_size_scratch);
            printf ("[Constraints][4] ERROR!\n");
#endif
            return -1;
        }
#ifdef DEBUG_CONSTRAINTS
        else
        {
            printf ("(Q) |C'| + |C''| <= |S|?, %'12.4f MB <= %'12.4f MB\n", mem_size_c_slice * 2, mem_size_scratch);
            printf ("[Constraints][4] PASSED!\n");
        }
#endif
    }
    
    // (5) Let D' be max(A',B') and E' be min(A',B'). Then, D' should be transposed firstly and then E' should be transposed secondly.
    //     Thus, E' + E'' <= S - A'
    if (mem_size_a_slice >= mem_size_b_slice)   // If A' >= B', A' should be first.
    {
        if (2 * mem_size_b_slice > mem_size_scratch - mem_size_a_slice)
        {
#ifdef DEBUG_CONSTRAINTS
            printf ("After A' is transposed, there is not enough space to transpose B'\n");
            printf ("[Constraints][5] ERROR!\n");
#endif
            return -1;
        }
#ifdef DEBUG_CONSTRAINTS
        else
        {
            printf ("If A' >= B', %'12.4f MB is enought to tranpose B', which requires %'12.4f MB\n", mem_size_scratch - mem_size_a_slice, 2 * mem_size_b_slice);
            printf ("[Constraints][5] PASSED!\n");
        }
#endif  
    }
    else                                        // If A' <= B', B' should be first.
    {
        if (2 * mem_size_a_slice > mem_size_scratch - mem_size_b_slice)
        {
#ifdef DEBUG_CONSTRAINTS
            printf ("After B' is transposed, there is not enough space to transpose A'\n");
            printf ("[Constraints][5] ERROR!\n");
#endif
            return -1;
        }
#ifdef DEBUG_CONSTRAINTS
        else
        {
            printf ("If A' <= B', %'12.4f MB is enought to tranpose A', which requires %'12.4f MB\n", mem_size_scratch - mem_size_b_slice, 2 * mem_size_a_slice);
            printf ("[Constraints][5] PASSED!\n");
        }
#endif  
    }
#ifdef DEBUG_CONSTRAINTS
    printf ("===========================================================================\n");
#endif

    return 1;
}