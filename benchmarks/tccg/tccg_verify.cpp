#include "tccg_verify.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "tccg_utils.hpp"

namespace {
thread_local VerificationTolerance g_verification_tolerance{1e-11, 1e-9};
thread_local VerificationResult g_verification_result{false, 0, 0, 0, 0};
VerificationResult check_correctness_comparison(int total_size, double* output_host, double* output_device);
void check_correctness_tccg_00(double* output, double* input_left, double* input_right, 
                            double* dev_output, const TconT::TCEquation& eq)
{
    int size_a = extent_of(eq, 'a');
    int size_b = extent_of(eq, 'b');
    int size_c = extent_of(eq, 'c');
    int size_d = extent_of(eq, 'd');

    // a(384),b(384),c(24)-b(384),d(384),a(394)-d(384),c(24)
    #pragma omp parallel
    #pragma omp for collapse(3)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    {
        for (int idx_d = 0; idx_d < size_d; idx_d++)
        {
            output[idx_a + (idx_b + (idx_c) * size_b) * size_a] += 
            input_left[idx_c + (idx_d) * size_c] * 
            input_right[idx_d + (idx_a + (idx_b) * size_a) * size_d];
        }
    }
    
    //
    int total_size  = size_a * size_b * size_c;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #01
// abc-bda-dc a:312;c:24;b:312;d:312;
void check_correctness_tccg_01(double* output, double* input_left, double* input_right, 
                            double* dev_output, const TconT::TCEquation& eq)
{    
    int size_a = extent_of(eq, 'a');
    int size_b = extent_of(eq, 'b');
    int size_c = extent_of(eq, 'c');
    int size_d = extent_of(eq, 'd');

    // a(384),b(384),c(24)-b(384),d(384),a(394)-d(384),c(24)
    #pragma omp parallel
    #pragma omp for collapse(3)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    {
        for (int idx_d = 0; idx_d < size_d; idx_d++)
        {
            output[idx_a + (idx_b + (idx_c) * size_b) * size_a] += 
            input_left[idx_b + (idx_d + (idx_a) * size_d) * size_b] * 
            input_right[idx_d + (idx_c) * size_d];
        }
    }

    //
    int total_size  = size_a * size_b * size_c;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #02
// abc-dca-bd a:312;c:296;b:24;d:312;
void check_correctness_tccg_02(double* output, double* input_left, double* input_right, 
                            double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    
    #pragma omp parallel
    #pragma omp for collapse(3)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    {
        for (int idx_d = 0; idx_d < size_d; idx_d++)
        {
            output[idx_a + (idx_b + (idx_c) * size_b) * size_a] += 
            input_left[idx_d + (idx_c + (idx_a) * size_c) * size_d] * 
            input_right[idx_b + (idx_d) * size_b];
        }
    }

    //
    int     total_size  = size_a * size_b * size_c;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #03
// abcd-dbea-ec a:72;c:24;b:72;e:72;d:72;
void check_correctness_tccg_03(double* output, double* input_left, double* input_right, 
                            double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    
    #pragma omp parallel
    #pragma omp for collapse(4)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    {
        for (int idx_e = 0; idx_e < size_e; idx_e++)
        {
            output[idx_a + (idx_b + (idx_c + (idx_d) * size_c) * size_b) * size_a] += 
            input_left[idx_d + (idx_b + (idx_e + (idx_a) * size_e) * size_b) * size_d] * 
            input_right[idx_e + (idx_c) * size_e];
        }
    }

    //
    int     total_size  = size_a * size_b * size_c * size_d;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #04
// abcd-deca-be a:72;c:72;b:24;e:72;d:72;
void check_correctness_tccg_04(double* output, double* input_left, double* input_right, 
                            double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    
    #pragma omp parallel
    #pragma omp for collapse(4)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    {
        for (int idx_e = 0; idx_e < size_e; idx_e++)
        {
            output[idx_a + (idx_b + (idx_c + (idx_d) * size_c) * size_b) * size_a] += 
            input_left[idx_d + (idx_e + (idx_c + (idx_a) * size_c) * size_e) * size_d] * 
            input_right[idx_b + (idx_e) *  size_b];
        }
    }

    //
    int     total_size  = size_a * size_b * size_c * size_d;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #05
// abcd-ebad-ce a:72;c:24;b:72;e:72;d:72;
void check_correctness_tccg_05(double* output, double* input_left, double* input_right, 
                            double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    
    #pragma omp parallel
    #pragma omp for collapse(4)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    {
        for (int idx_e = 0; idx_e < size_e; idx_e++)
        {
            output[idx_a + (idx_b + (idx_c + (idx_d) * size_c) * size_b) * size_a] += 
            input_left[idx_e + (idx_b + (idx_a + (idx_d) * size_a) * size_b) * size_e] * 
            input_right[idx_c + (idx_e) * size_c];
        }
    }

    //
    int     total_size  = size_a * size_b * size_c * size_d;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #06
// abcde-efbad-cf a:48;c:24;b:32;e:48;d:32;f:32;
void check_correctness_tccg_06(double* output, double* input_left, double* input_right, 
                            double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    
    #pragma omp parallel
    #pragma omp for collapse(5)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    {
        for (int idx_f = 0; idx_f < size_f; idx_f++)
        {
            output[idx_a + (idx_b + (idx_c + (idx_d + (idx_e) * size_d) * size_c) * size_b) * size_a] += 
            input_left[idx_e+ (idx_f + (idx_b + (idx_a + (idx_d) * size_a) * size_b) * size_f) * size_e] * 
            input_right[idx_c + (idx_f) * size_c];
        }
    }

    //
    int     total_size  = size_a * size_b * size_c * size_d * size_e;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #07
// abcde-ecbfa-fd a:48;c:32;b:32;e:48;d:24;f:48;
void check_correctness_tccg_07(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    
    #pragma omp parallel
    #pragma omp for collapse(5)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    {
        for (int idx_f = 0; idx_f < size_f; idx_f++)
        {
            output      [idx_a + (idx_b + (idx_c + (idx_d + (idx_e) * size_d) * size_c) * size_b) * size_a] += 
            input_left  [idx_e + (idx_c + (idx_b + (idx_f + (idx_a) * size_f) * size_b) * size_c) * size_e] * 
            input_right [idx_f + (idx_d) * size_f];
        }
    }

    //
    int total_size  = size_a * size_b * size_c * size_d * size_e;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #08
// abcde-efcad-bf a:48;c:32;b:24;e:48;d:32;f:32;
void check_correctness_tccg_08(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    
    #pragma omp parallel
    #pragma omp for collapse(5)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    {
        for (int idx_f = 0; idx_f < size_f; idx_f++)
        {   // abcde-efcad-bf a:48;c:32;b:24;e:48;d:32;f:32;
            output      [idx_a + (idx_b + (idx_c + (idx_d + (idx_e) * size_d) * size_c) * size_b) * size_a] += 
            input_left  [idx_e + (idx_f + (idx_c + (idx_a + (idx_d) * size_a) * size_c) * size_f) * size_e] * 
            input_right [idx_b + (idx_f) * size_b];
        }
    }

    //
    int     total_size  = size_a * size_b * size_c * size_d * size_e;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #09
// abcd-ea-ebcd a:72;c:72;b:72;e:72;d:72;
void check_correctness_tccg_09(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    
    #pragma omp parallel
    #pragma omp for collapse(4)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    {
        for (int idx_e = 0; idx_e < size_e; idx_e++)
        {
            output      [idx_a + (idx_b + (idx_c + (idx_d) * size_c) * size_b) * size_a] += 
            input_left  [idx_e + (idx_a) * size_e] * 
            input_right [idx_e + (idx_b + (idx_c + (idx_d) * size_c) * size_b) * size_e];
        }
    }

    //
    int total_size  = size_a * size_b * size_c * size_d;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #10
// abcd-eb-aecd a:72;c:72;b:72;e:72;d:72;
void check_correctness_tccg_10(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    
    #pragma omp parallel
    #pragma omp for collapse(4)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    {
        for (int idx_e = 0; idx_e < size_e; idx_e++)
        {
            output      [idx_a + (idx_b + (idx_c + (idx_d) * size_c) * size_b) * size_a] += 
            input_left  [idx_e + (idx_b) * size_e] * 
            input_right [idx_a + (idx_e + (idx_c + (idx_d) * size_c) * size_e) * size_a];
        }
    }

    //
    int total_size  = size_a * size_b * size_c * size_d;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #11
// abcd-ec-abed a:72;c:72;b:72;e:72;d:72;
void check_correctness_tccg_11(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    
    #pragma omp parallel
    #pragma omp for collapse(4)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    {
        for (int idx_e = 0; idx_e < size_e; idx_e++)
        {
            output      [idx_a + (idx_b + (idx_c + (idx_d) * size_c) * size_b) * size_a] += 
            input_left  [idx_e + (idx_c) * size_e] * 
            input_right [idx_a + (idx_b + (idx_e + (idx_d) * size_e) * size_b) * size_a];
        }
    }

    //
    int total_size  = size_a * size_b * size_c * size_d;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #12
// ab-ac-cb a:5136;c:5136;b:5120;
void check_correctness_tccg_12(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    
    #pragma omp parallel
    #pragma omp for collapse(2)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    {
        for (int idx_c = 0; idx_c < size_c; idx_c++)
        {
            output      [idx_a + (idx_b) * size_a] += 
            input_left  [idx_a + (idx_c) * size_a] * 
            input_right [idx_c + (idx_b) * size_c];
        }
    }

    //
    int total_size  = size_a * size_b;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #13
// ab-acd-dbc a:312;c:296;b:296;d:312;
void check_correctness_tccg_13(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    
    #pragma omp parallel
    #pragma omp for collapse(2)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    {
        for (int idx_d = 0; idx_d < size_d; idx_d++)
        {
            output      [idx_a + (idx_b) * size_a] += 
            input_left  [idx_a + (idx_c + (idx_d) * size_c) * size_a] * 
            input_right [idx_d + (idx_b + (idx_c) * size_b) * size_d];
        }
    }

    //
    int total_size  = size_a * size_b;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #14
// ab-cad-dcb a:312;c:312;b:296;d:312;
void check_correctness_tccg_14(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');

    long long int total_op = 0;    

    #pragma omp parallel
    #pragma omp for collapse(2)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    {
        for (int idx_d = 0; idx_d < size_d; idx_d++)
        {
            output      [idx_a + (idx_b) * size_a] += 
            input_left  [idx_c + (idx_a + (idx_d) * size_a) * size_c] * 
            input_right [idx_d + (idx_c + (idx_b) * size_c) * size_d];
	        total_op++;
        }
    }

    //
    int total_size  = size_a * size_b;
    printf ("# of Operations: %lld\n", total_op * 2);
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #15
// abc-acd-db a:312;c:296;b:296;d:312;
void check_correctness_tccg_15(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    
    #pragma omp parallel
    #pragma omp for collapse(3)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    {
        for (int idx_d = 0; idx_d < size_d; idx_d++)
        {
            output      [idx_a + (idx_b + (idx_c) * size_b) * size_a] += 
            input_left  [idx_a + (idx_c + (idx_d) * size_c) * size_a] * 
            input_right [idx_d + (idx_b) * size_d];
        }
    }

    //
    int total_size  = size_a * size_b * size_c;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #16
// abc-ad-bdc a:312;c:296;b:312;d:296;
void check_correctness_tccg_16(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    
    #pragma omp parallel
    #pragma omp for collapse(3)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    {
        for (int idx_d = 0; idx_d < size_d; idx_d++)
        {
            output      [idx_a + (idx_b + (idx_c) * size_b) * size_a] += 
            input_left  [idx_a + (idx_d) * size_a] * 
            input_right [idx_b + (idx_d + (idx_c) * size_d) * size_b];
        }
    }

    //
    int total_size  = size_a * size_b * size_c;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #17
// abc-adc-bd a:312;c:296;b:312;d:296;
void check_correctness_tccg_17(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    
    #pragma omp parallel
    #pragma omp for collapse(3)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    {
        for (int idx_d = 0; idx_d < size_d; idx_d++)
        {
            output      [idx_a + (idx_b + (idx_c) * size_b) * size_a] += 
            input_left  [idx_a + (idx_d + (idx_c) * size_d) * size_a] * 
            input_right [idx_b + (idx_d) * size_b];
        }
    }

    //
    int total_size  = size_a * size_b * size_c;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #18
// abc-adc-db a:312;c:296;b:296;d:312; **
void check_correctness_tccg_18(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    
    #pragma omp parallel
    #pragma omp for collapse(3)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    {
        for (int idx_d = 0; idx_d < size_d; idx_d++)
        {
            output      [idx_a + (idx_b + (idx_c) * size_b) * size_a] += 
            input_left  [idx_a + (idx_d + (idx_c) * size_d) * size_a] * 
            input_right [idx_d + (idx_b) * size_d];
        }
    }

    //
    int total_size  = size_a * size_b * size_c;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #19
// abc-adec-ebd a:72;c:72;b:72;e:72;d:72; **
void check_correctness_tccg_19(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    
    #pragma omp parallel
    #pragma omp for collapse(3)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    {
        for (int idx_e = 0; idx_e < size_e; idx_e++)
        {
            output     [idx_a + (idx_b + (idx_c) * size_b) * size_a] += 
            input_left [idx_a + (idx_d + (idx_e + (idx_c) * size_e) * size_d) * size_a] * 
            input_right[idx_e + (idx_b + (idx_d) * size_b) * size_e];
        }
    }

    //
    int total_size = size_a * size_b * size_c;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #20
// abcd-aebf-dfce a:72;c:72;b:72;e:72;d:72;f:72;
void check_correctness_tccg_20(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
       
    #pragma omp parallel
    #pragma omp for collapse(4)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    {
        for (int idx_f = 0; idx_f < size_f; idx_f++)
        {
            output      [idx_a + (idx_b + (idx_c + (idx_d) * size_c) * size_b) * size_a] += 
            input_left  [idx_a + (idx_e + (idx_b + (idx_f) * size_b) * size_e) * size_a] * 
            input_right [idx_d + (idx_f + (idx_c + (idx_e) * size_c) * size_f) * size_d];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #21
// abcd-aebf-fdec a:72;c:72;b:72;e:72;d:72;f:72;
void check_correctness_tccg_21(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    
    #pragma omp parallel
    #pragma omp for collapse(4)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    {
        for (int idx_f = 0; idx_f < size_f; idx_f++)
        {
            output      [idx_a + (idx_b + (idx_c + (idx_d) * size_c) * size_b) * size_a] += 
            input_left  [idx_a + (idx_e + (idx_b + (idx_f) * size_b) * size_e) * size_a] * 
            input_right [idx_f + (idx_d + (idx_e + (idx_c) * size_e) * size_d) * size_f];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #22
// abcd-aecf-bfde a:72;c:72;b:72;e:72;d:72;f:72;
void check_correctness_tccg_22(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    
    #pragma omp parallel
    #pragma omp for collapse(4)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    {
        for (int idx_f = 0; idx_f < size_f; idx_f++)
        {
            output      [idx_a + (idx_b + (idx_c + (idx_d) * size_c) * size_b) * size_a] += 
            input_left  [idx_a + (idx_e + (idx_c + (idx_f) * size_c) * size_e) * size_a] * 
            input_right [idx_b + (idx_f + (idx_d + (idx_e) * size_d) * size_f) * size_b];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #23
// abcd-aecf-fbed a:72;c:72;b:72;e:72;d:72;f:72;
void check_correctness_tccg_23(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    
    #pragma omp parallel
    #pragma omp for collapse(4)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    {
        for (int idx_f = 0; idx_f < size_f; idx_f++)
        {
            output      [idx_a + (idx_b + (idx_c + (idx_d) * size_c) * size_b) * size_a] += 
            input_left  [idx_a + (idx_e + (idx_c + (idx_f) * size_c) * size_e) * size_a] * 
            input_right [idx_f + (idx_b + (idx_e + (idx_d) * size_e) * size_b) * size_f];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #24
// abcd-aedf-bfce a:72;c:72;b:72;e:72;d:72;f:72;
void check_correctness_tccg_24(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
       
    #pragma omp parallel
    #pragma omp for collapse(4)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    {
        for (int idx_f = 0; idx_f < size_f; idx_f++)
        {
            output      [idx_a + (idx_b + (idx_c + (idx_d) * size_c) * size_b) * size_a] += 
            input_left  [idx_a + (idx_e + (idx_d + (idx_f) * size_d) * size_e) * size_a] * 
            input_right [idx_b + (idx_f + (idx_c + (idx_e) * size_c) * size_f) * size_b];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #25
// abcd-aedf-fbec a:72;c:72;b:72;e:72;d:72;f:72;
void check_correctness_tccg_25(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    
    #pragma omp parallel
    #pragma omp for collapse(4)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    {
        for (int idx_f = 0; idx_f < size_f; idx_f++)
        {
            output      [idx_a + (idx_b + (idx_c + (idx_d) * size_c) * size_b) * size_a] += 
            input_left  [idx_a + (idx_e + (idx_d + (idx_f) * size_d) * size_e) * size_a] * 
            input_right [idx_f + (idx_b + (idx_e + (idx_c) * size_e) * size_b) * size_f];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #26
// abcd-aefb-fdce a:72;c:72;b:72;e:72;d:72;f:72;
void check_correctness_tccg_26(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    
    #pragma omp parallel
    #pragma omp for collapse(4)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    {
        for (int idx_f = 0; idx_f < size_f; idx_f++)
        {
            output      [idx_a + (idx_b + (idx_c + (idx_d) * size_c) * size_b) * size_a] += 
            input_left  [idx_a + (idx_e + (idx_f + (idx_b) * size_f) * size_e) * size_a] * 
            input_right [idx_f + (idx_d + (idx_c + (idx_e) * size_c) * size_d) * size_f];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #27
// abcd-aefc-fbed a:72;c:72;b:72;e:72;d:72;f:72;
void check_correctness_tccg_27(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    
    #pragma omp parallel
    #pragma omp for collapse(4)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    {
        for (int idx_f = 0; idx_f < size_f; idx_f++)
        {
            output      [idx_a + (idx_b + (idx_c + (idx_d) * size_c) * size_b) * size_a] += 
            input_left  [idx_a + (idx_e + (idx_f + (idx_c) * size_f) * size_e) * size_a] * 
            input_right [idx_f + (idx_b + (idx_e + (idx_d) * size_e) * size_b) * size_f];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #28
// abcd-eafb-fdec a:72;c:72;b:72;e:72;d:72;f:72;
void check_correctness_tccg_28(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    
    #pragma omp parallel
    #pragma omp for collapse(4)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    {
        for (int idx_f = 0; idx_f < size_f; idx_f++)
        {
            output      [idx_a + (idx_b + (idx_c + (idx_d) * size_c) * size_b) * size_a] += 
            input_left  [idx_e + (idx_a + (idx_f + (idx_b) * size_f) * size_a) * size_e] * 
            input_right [idx_f + (idx_d + (idx_e + (idx_c) * size_e) * size_d) * size_f];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #29
// abcd-eafc-bfde a:72;c:72;b:72;e:72;d:72;f:72;
void check_correctness_tccg_29(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    
    #pragma omp parallel
    #pragma omp for collapse(4)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    {
        for (int idx_f = 0; idx_f < size_f; idx_f++)
        {
            output      [idx_a + (idx_b + (idx_c + (idx_d) * size_c) * size_b) * size_a] += 
            input_left  [idx_e + (idx_a + (idx_f + (idx_c) * size_f) * size_a) * size_e] * 
            input_right [idx_b + (idx_f + (idx_d + (idx_e) * size_d) * size_f) * size_b];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #30
// abcd-eafd-fbec a:72;c:72;b:72;e:72;d:72;f:72;
void check_correctness_tccg_30(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    
    #pragma omp parallel
    #pragma omp for collapse(4)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    {
        for (int idx_f = 0; idx_f < size_f; idx_f++)
        {
            output      [idx_a + (idx_b + (idx_c + (idx_d) * size_c) * size_b) * size_a] += 
            input_left  [idx_e + (idx_a + (idx_f + (idx_d) * size_f) * size_a) * size_e] * 
            input_right [idx_f + (idx_b + (idx_e + (idx_c) * size_e) * size_b) * size_f];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #31
// abcdef-dega-gfbc a:24;c:16;b:16;e:16;d:24;g:24;f:16;
void check_correctness_tccg_31(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    int size_g  = extent_of(eq, 'g');
    
    #pragma omp parallel
    #pragma omp for collapse(6)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    for (int idx_f = 0; idx_f < size_f; idx_f++)
    {
        for (int idx_g = 0; idx_g < size_g; idx_g++)
        {
            output     [idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a] +=
            input_left [idx_d + (idx_e + (idx_g + (idx_a) * size_g) * size_e) * size_d] * 
            input_right[idx_g + (idx_f + (idx_b + (idx_c) * size_b) * size_f) * size_g];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d * size_e * size_f;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #32
// abcdef-degb-gfac a:24;c:16;b:16;e:16;d:24;g:24;f:16;
void check_correctness_tccg_32(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    int size_g  = extent_of(eq, 'g');
    
    #pragma omp parallel
    #pragma omp for collapse(6)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    for (int idx_f = 0; idx_f < size_f; idx_f++)
    {
        for (int idx_g = 0; idx_g < size_g; idx_g++)
        {
            output     [idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a] +=
            input_left [idx_d + (idx_e + (idx_g + (idx_b) * size_g) * size_e) * size_d] * 
            input_right[idx_g + (idx_f + (idx_a + (idx_c) * size_a) * size_f) * size_g];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d * size_e * size_f;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #33
// abcdef-degc-gfab a:24;c:16;b:16;e:16;d:24;g:24;f:16;
void check_correctness_tccg_33(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    int size_g  = extent_of(eq, 'g');
    
    #pragma omp parallel
    #pragma omp for collapse(6)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    for (int idx_f = 0; idx_f < size_f; idx_f++)
    {
        for (int idx_g = 0; idx_g < size_g; idx_g++)
        {
            output     [idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a] +=
            input_left [idx_d + (idx_e + (idx_g + (idx_c) * size_g) * size_e) * size_d] * 
            input_right[idx_g + (idx_f + (idx_a + (idx_b) * size_a) * size_f) * size_g];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d * size_e * size_f;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #34
// abcdef-dfga-gebc a:24;c:16;b:16;e:16;d:24;g:24;f:16;
void check_correctness_tccg_34(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    int size_g  = extent_of(eq, 'g');
    
    #pragma omp parallel
    #pragma omp for collapse(6)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    for (int idx_f = 0; idx_f < size_f; idx_f++)
    {
        for (int idx_g = 0; idx_g < size_g; idx_g++)
        {
            output     [idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a] +=
            input_left [idx_d + (idx_f + (idx_g + (idx_a) * size_g) * size_f) * size_d] * 
            input_right[idx_g + (idx_e + (idx_b + (idx_c) * size_b) * size_e) * size_g];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d * size_e * size_f;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #35
// abcdef-dfgb-geac a:24;c:16;b:16;e:16;d:24;g:24;f:16;
void check_correctness_tccg_35(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    int size_g  = extent_of(eq, 'g');
    
    #pragma omp parallel
    #pragma omp for collapse(6)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    for (int idx_f = 0; idx_f < size_f; idx_f++)
    {
        for (int idx_g = 0; idx_g < size_g; idx_g++)
        {
            output     [idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a] +=
            input_left [idx_d + (idx_f + (idx_g + (idx_b) * size_g) * size_f) * size_d] * 
            input_right[idx_g + (idx_e + (idx_a + (idx_c) * size_a) * size_e) * size_g];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d * size_e * size_f;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #36
// abcdef-dfgc-geab a:24;c:16;b:16;e:16;d:24;g:24;f:16;
void check_correctness_tccg_36(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    int size_g  = extent_of(eq, 'g');
    
    #pragma omp parallel
    #pragma omp for collapse(6)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    for (int idx_f = 0; idx_f < size_f; idx_f++)
    {
        for (int idx_g = 0; idx_g < size_g; idx_g++)
        {
            output     [idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a] +=
            input_left [idx_d + (idx_f + (idx_g + (idx_c) * size_g) * size_f) * size_d] * 
            input_right[idx_g + (idx_e + (idx_a + (idx_b) * size_a) * size_e) * size_g];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d * size_e * size_f;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #37
// abcdef-efga-gdbc a:24;c:16;b:16;e:24;d:16;g:24;f:16;
void check_correctness_tccg_37(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    int size_g  = extent_of(eq, 'g');
    
    #pragma omp parallel
    #pragma omp for collapse(6)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    for (int idx_f = 0; idx_f < size_f; idx_f++)
    {
        for (int idx_g = 0; idx_g < size_g; idx_g++)
        {
            output     [idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a] +=
            input_left [idx_e + (idx_f + (idx_g + (idx_a) * size_g) * size_f) * size_e] * 
            input_right[idx_g + (idx_d + (idx_b + (idx_c) * size_b) * size_d) * size_g];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d * size_e * size_f;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #38
// abcdef-efgb-gdac a:24;c:16;b:16;e:24;d:16;g:24;f:16;
void check_correctness_tccg_38(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    int size_g  = extent_of(eq, 'g');
    
    #pragma omp parallel
    #pragma omp for collapse(6)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    for (int idx_f = 0; idx_f < size_f; idx_f++)
    {
        for (int idx_g = 0; idx_g < size_g; idx_g++)
        {
            output     [idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a] +=
            input_left [idx_e + (idx_f + (idx_g + (idx_b) * size_g) * size_f) * size_e] * 
            input_right[idx_g + (idx_d + (idx_a + (idx_c) * size_a) * size_d) * size_g];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d * size_e * size_f;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #39
// abcdef-efgc-gdab a:24;c:16;b:16;e:24;d:16;g:24;f:16;
void check_correctness_tccg_39(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    int size_g  = extent_of(eq, 'g');
    
    #pragma omp parallel
    #pragma omp for collapse(6)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    for (int idx_f = 0; idx_f < size_f; idx_f++)
    {
        for (int idx_g = 0; idx_g < size_g; idx_g++)
        {
            output     [idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a] +=
            input_left [idx_e + (idx_f + (idx_g + (idx_c) * size_g) * size_f) * size_e] * 
            input_right[idx_g + (idx_d + (idx_a + (idx_b) * size_a) * size_d) * size_g];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d * size_e * size_f;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #40
// abcdef-gdab-efgc a:24;c:16;b:16;e:24;d:16;g:24;f:16;
void check_correctness_tccg_40(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    int size_g  = extent_of(eq, 'g');
    
    #pragma omp parallel
    #pragma omp for collapse(6)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    for (int idx_f = 0; idx_f < size_f; idx_f++)
    {
        for (int idx_g = 0; idx_g < size_g; idx_g++)
        {
            output     [idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a] +=
            input_left [idx_g + (idx_d + (idx_a + (idx_b) * size_a) * size_d) * size_g] * 
            input_right[idx_e + (idx_f + (idx_g + (idx_c) * size_g) * size_f) * size_e];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d * size_e * size_f;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #41
// abcdef-gdac-efgb a:24;c:16;b:16;e:24;d:16;g:24;f:16;
void check_correctness_tccg_41(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    int size_g  = extent_of(eq, 'g');
    
    #pragma omp parallel
    #pragma omp for collapse(6)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    for (int idx_f = 0; idx_f < size_f; idx_f++)
    {
        for (int idx_g = 0; idx_g < size_g; idx_g++)
        {
            output     [idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a] +=
            input_left [idx_g + (idx_d + (idx_a + (idx_c) * size_a) * size_d) * size_g] * 
            input_right[idx_e + (idx_f + (idx_g + (idx_b) * size_g) * size_f) * size_e];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d * size_e * size_f;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #42
// abcdef-gdbc-efga a:24;c:16;b:16;e:24;d:16;g:24;f:16;
void check_correctness_tccg_42(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    int size_g  = extent_of(eq, 'g');
    
    #pragma omp parallel
    #pragma omp for collapse(6)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    for (int idx_f = 0; idx_f < size_f; idx_f++)
    {
        for (int idx_g = 0; idx_g < size_g; idx_g++)
        {
            output     [idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a] +=
            input_left [idx_g + (idx_d + (idx_b + (idx_c) * size_b) * size_d) * size_g] * 
            input_right[idx_e + (idx_f + (idx_g + (idx_a) * size_g) * size_f) * size_e];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d * size_e * size_f;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #43
// abcdef-geab-dfgc a:24;c:16;b:16;e:16;d:24;g:24;f:16;
void check_correctness_tccg_43(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    int size_g  = extent_of(eq, 'g');
    
    #pragma omp parallel
    #pragma omp for collapse(6)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    for (int idx_f = 0; idx_f < size_f; idx_f++)
    {
        for (int idx_g = 0; idx_g < size_g; idx_g++)
        {
            output     [idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a] +=
            input_left [idx_g + (idx_e + (idx_a + (idx_b) * size_a) * size_e) * size_g] * 
            input_right[idx_d + (idx_f + (idx_g + (idx_c) * size_g) * size_f) * size_d];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d * size_e * size_f;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #44
// abcdef-geac-dfgb a:24;c:16;b:16;e:16;d:24;g:24;f:16;
void check_correctness_tccg_44(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    int size_g  = extent_of(eq, 'g');
    
    #pragma omp parallel
    #pragma omp for collapse(6)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    for (int idx_f = 0; idx_f < size_f; idx_f++)
    {
        for (int idx_g = 0; idx_g < size_g; idx_g++)
        {
            output     [idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a] +=
            input_left [idx_g + (idx_e + (idx_a + (idx_c) * size_a) * size_e) * size_g] * 
            input_right[idx_d + (idx_f + (idx_g + (idx_b) * size_g) * size_f) * size_d];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d * size_e * size_f;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #45
// abcdef-gebc-dfga a:24;c:16;b:16;e:16;d:24;g:24;f:16;
void check_correctness_tccg_45(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    int size_g  = extent_of(eq, 'g');
    
    #pragma omp parallel
    #pragma omp for collapse(6)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    for (int idx_f = 0; idx_f < size_f; idx_f++)
    {
        for (int idx_g = 0; idx_g < size_g; idx_g++)
        {
            output     [idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a] +=
            input_left [idx_g + (idx_e + (idx_b + (idx_c) * size_b) * size_e) * size_g] * 
            input_right[idx_d + (idx_f + (idx_g + (idx_a) * size_g) * size_f) * size_d];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d * size_e * size_f;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #46
// abcdef-gfab-degc a:24;c:16;b:16;e:16;d:24;g:24;f:16;
void check_correctness_tccg_46(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    int size_g  = extent_of(eq, 'g');
    
    #pragma omp parallel
    #pragma omp for collapse(6)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    for (int idx_f = 0; idx_f < size_f; idx_f++)
    {
        for (int idx_g = 0; idx_g < size_g; idx_g++)
        {
            output     [idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a] +=
            input_left [idx_g + (idx_f + (idx_a + (idx_b) * size_a) * size_f) * size_g] * 
            input_right[idx_d + (idx_e + (idx_g + (idx_c) * size_g) * size_e) * size_d];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d * size_e * size_f;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #47
// abcdef-gfac-degb a:24;c:16;b:16;e:16;d:24;g:24;f:16;
void check_correctness_tccg_47(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    int size_g  = extent_of(eq, 'g');
    
    #pragma omp parallel
    #pragma omp for collapse(6)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    for (int idx_f = 0; idx_f < size_f; idx_f++)
    {
        for (int idx_g = 0; idx_g < size_g; idx_g++)
        {
            output     [idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a] +=
            input_left [idx_g + (idx_f + (idx_a + (idx_c) * size_a) * size_f) * size_g] * 
            input_right[idx_d + (idx_e + (idx_g + (idx_b) * size_g) * size_e) * size_d];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d * size_e * size_f;
    check_correctness_comparison(total_size, output, dev_output);
}

// tccg #48
// abcdef-gfbc-dega a:24;c:16;b:16;e:16;d:24;g:24;f:16;
void check_correctness_tccg_48(double* output, double* input_left, double* input_right, 
    double* dev_output, const TconT::TCEquation& eq)
{
    int size_a  = extent_of(eq, 'a');
    int size_b  = extent_of(eq, 'b');
    int size_c  = extent_of(eq, 'c');
    int size_d  = extent_of(eq, 'd');
    int size_e  = extent_of(eq, 'e');
    int size_f  = extent_of(eq, 'f');
    int size_g  = extent_of(eq, 'g');
    
    #pragma omp parallel
    #pragma omp for collapse(6)
    for (int idx_a = 0; idx_a < size_a; idx_a++)
    for (int idx_b = 0; idx_b < size_b; idx_b++)
    for (int idx_c = 0; idx_c < size_c; idx_c++)
    for (int idx_d = 0; idx_d < size_d; idx_d++)
    for (int idx_e = 0; idx_e < size_e; idx_e++)
    for (int idx_f = 0; idx_f < size_f; idx_f++)
    {
        for (int idx_g = 0; idx_g < size_g; idx_g++)
        {
            output     [idx_a + (idx_b + (idx_c + (idx_d + (idx_e + (idx_f) * size_e) * size_d) * size_c) * size_b) * size_a] +=
            input_left [idx_g + (idx_f + (idx_b + (idx_c) * size_b) * size_f) * size_g] * 
            input_right[idx_d + (idx_e + (idx_g + (idx_a) * size_g) * size_e) * size_d];
        }
    }

    //
    int total_size = size_a * size_b * size_c * size_d * size_e * size_f;
    check_correctness_comparison(total_size, output, dev_output);
}


//
VerificationResult check_correctness_comparison(int total_size, double* output_host, double* output_device)
{
    if (total_size < 0) {
        fprintf(stderr, "Invalid total size: %d\n", total_size);
        return VerificationResult{false, 0, total_size, 0, 0};
    }
    if (!output_host || !output_device) {
        fprintf(stderr, "Null pointer detected for output arrays.\n");
        return VerificationResult{false, 0, total_size, 0, 0};
    }
    printf ("=======================================================================\n");
#ifdef _OPENMP
    printf("OpenMP enabled. max threads = %d\n", omp_get_max_threads());
#else
    printf("OpenMP NOT enabled.\n");
#endif
    
    // Tolerances:
    //  - abs_tol handles values near 0
    //  - rel_tol scales with magnitude for larger values
    const double abs_tol = g_verification_tolerance.abs_tol;
    const double rel_tol = g_verification_tolerance.rel_tol;
    
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
                printf ("[%d/%d] h=%0.4e, d=%0.4e, |err|=%.03e, tol=%0.3e\n", 
                    i, total_size, h, d, err, tol);
            }
        } else {
            same++;
        }

    }
    printf ("Differences: %d / %d (Same: %d, Non-finite: %d)\n", 
        diff, total_size, same, non_finite);
    printf ("=======================================================================\n");
    g_verification_result = VerificationResult{diff == 0, diff, total_size, same, non_finite};
    return g_verification_result;
}
}  // namespace

VerificationTolerance verification_tolerance_for(TconT::ScalarType scalar_type)
{
    switch (scalar_type) {
        case TconT::ScalarType::Float32:
            // TF32 typically keeps FP32 range with a 10-bit mantissa, so we allow
            // noticeably looser relative error than true FP32 GEMM validation.
            return VerificationTolerance{2e-3, 5e-2};
        case TconT::ScalarType::Float64:
            return VerificationTolerance{1e-11, 1e-9};
    }
    return VerificationTolerance{1e-11, 1e-9};
}

namespace {
VerificationResult verify_tccg_case_reference(
    size_t case_index,
    const TconT::TCEquation& eq,
    double* output_reference,
    double* output_device,
    double* input_left,
    double* input_right)
{
    g_verification_result = VerificationResult{false, 0, 0, 0, 0};
    switch (case_index) {
        case 0: check_correctness_tccg_00(output_reference, input_left, input_right, output_device, eq); break;
        case 1: check_correctness_tccg_01(output_reference, input_left, input_right, output_device, eq); break;
        case 2: check_correctness_tccg_02(output_reference, input_left, input_right, output_device, eq); break;
        case 3: check_correctness_tccg_03(output_reference, input_left, input_right, output_device, eq); break;
        case 4: check_correctness_tccg_04(output_reference, input_left, input_right, output_device, eq); break;
        case 5: check_correctness_tccg_05(output_reference, input_left, input_right, output_device, eq); break;
        case 6: check_correctness_tccg_06(output_reference, input_left, input_right, output_device, eq); break;
        case 7: check_correctness_tccg_07(output_reference, input_left, input_right, output_device, eq); break;
        case 8: check_correctness_tccg_08(output_reference, input_left, input_right, output_device, eq); break;
        case 9: check_correctness_tccg_09(output_reference, input_left, input_right, output_device, eq); break;
        case 10: check_correctness_tccg_10(output_reference, input_left, input_right, output_device, eq); break;
        case 11: check_correctness_tccg_11(output_reference, input_left, input_right, output_device, eq); break;
        case 12: check_correctness_tccg_12(output_reference, input_left, input_right, output_device, eq); break;
        case 13: check_correctness_tccg_13(output_reference, input_left, input_right, output_device, eq); break;
        case 14: check_correctness_tccg_14(output_reference, input_left, input_right, output_device, eq); break;
        case 15: check_correctness_tccg_15(output_reference, input_left, input_right, output_device, eq); break;
        case 16: check_correctness_tccg_16(output_reference, input_left, input_right, output_device, eq); break;
        case 17: check_correctness_tccg_17(output_reference, input_left, input_right, output_device, eq); break;
        case 18: check_correctness_tccg_18(output_reference, input_left, input_right, output_device, eq); break;
        case 19: check_correctness_tccg_19(output_reference, input_left, input_right, output_device, eq); break;
        case 20: check_correctness_tccg_20(output_reference, input_left, input_right, output_device, eq); break;
        case 21: check_correctness_tccg_21(output_reference, input_left, input_right, output_device, eq); break;
        case 22: check_correctness_tccg_22(output_reference, input_left, input_right, output_device, eq); break;
        case 23: check_correctness_tccg_23(output_reference, input_left, input_right, output_device, eq); break;
        case 24: check_correctness_tccg_24(output_reference, input_left, input_right, output_device, eq); break;
        case 25: check_correctness_tccg_25(output_reference, input_left, input_right, output_device, eq); break;
        case 26: check_correctness_tccg_26(output_reference, input_left, input_right, output_device, eq); break;
        case 27: check_correctness_tccg_27(output_reference, input_left, input_right, output_device, eq); break;
        case 28: check_correctness_tccg_28(output_reference, input_left, input_right, output_device, eq); break;
        case 29: check_correctness_tccg_29(output_reference, input_left, input_right, output_device, eq); break;
        case 30: check_correctness_tccg_30(output_reference, input_left, input_right, output_device, eq); break;
        case 31: check_correctness_tccg_31(output_reference, input_left, input_right, output_device, eq); break;
        case 32: check_correctness_tccg_32(output_reference, input_left, input_right, output_device, eq); break;
        case 33: check_correctness_tccg_33(output_reference, input_left, input_right, output_device, eq); break;
        case 34: check_correctness_tccg_34(output_reference, input_left, input_right, output_device, eq); break;
        case 35: check_correctness_tccg_35(output_reference, input_left, input_right, output_device, eq); break;
        case 36: check_correctness_tccg_36(output_reference, input_left, input_right, output_device, eq); break;
        case 37: check_correctness_tccg_37(output_reference, input_left, input_right, output_device, eq); break;
        case 38: check_correctness_tccg_38(output_reference, input_left, input_right, output_device, eq); break;
        case 39: check_correctness_tccg_39(output_reference, input_left, input_right, output_device, eq); break;
        case 40: check_correctness_tccg_40(output_reference, input_left, input_right, output_device, eq); break;
        case 41: check_correctness_tccg_41(output_reference, input_left, input_right, output_device, eq); break;
        case 42: check_correctness_tccg_42(output_reference, input_left, input_right, output_device, eq); break;
        case 43: check_correctness_tccg_43(output_reference, input_left, input_right, output_device, eq); break;
        case 44: check_correctness_tccg_44(output_reference, input_left, input_right, output_device, eq); break;
        case 45: check_correctness_tccg_45(output_reference, input_left, input_right, output_device, eq); break;
        case 46: check_correctness_tccg_46(output_reference, input_left, input_right, output_device, eq); break;
        case 47: check_correctness_tccg_47(output_reference, input_left, input_right, output_device, eq); break;
        case 48: check_correctness_tccg_48(output_reference, input_left, input_right, output_device, eq); break;
        default:
            throw std::out_of_range("Unsupported TCCG verification case index");
    }
    return g_verification_result;
}

template <typename SrcType>
std::vector<double> convert_to_double(const SrcType* src, size_t count)
{
    std::vector<double> out(count);
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<double>(src[i]);
    }
    return out;
}

VerificationResult verify_tccg_case_float(
    size_t case_index,
    const TconT::TCEquation& eq,
    const float* output_device,
    const float* input_left,
    const float* input_right)
{
    const auto output_size = static_cast<size_t>(compute_tensor_size(eq.modeC, eq.extent));
    const auto left_size = static_cast<size_t>(compute_tensor_size(eq.modeA, eq.extent));
    const auto right_size = static_cast<size_t>(compute_tensor_size(eq.modeB, eq.extent));

    std::vector<double> reference_output(output_size, 0.0);
    auto output = convert_to_double(output_device, output_size);
    auto left = convert_to_double(input_left, left_size);
    auto right = convert_to_double(input_right, right_size);

    return verify_tccg_case_reference(
        case_index,
        eq,
        reference_output.data(),
        output.data(),
        left.data(),
        right.data());
}
}  // namespace

VerificationResult verify_tccg_case(
    size_t case_index,
    const TconT::TCEquation& eq,
    const void* output_device,
    const void* input_left,
    const void* input_right)
{
    const auto output_size = static_cast<size_t>(compute_tensor_size(eq.modeC, eq.extent));
    const auto left_size = static_cast<size_t>(compute_tensor_size(eq.modeA, eq.extent));
    const auto right_size = static_cast<size_t>(compute_tensor_size(eq.modeB, eq.extent));

    g_verification_tolerance = verification_tolerance_for(eq.scalar_type);

    switch (eq.scalar_type) {
        case TconT::ScalarType::Float64: {
            std::vector<double> reference_output(output_size, 0.0);
            auto output = convert_to_double(static_cast<const double*>(output_device), output_size);
            auto left = convert_to_double(static_cast<const double*>(input_left), left_size);
            auto right = convert_to_double(static_cast<const double*>(input_right), right_size);
            return verify_tccg_case_reference(case_index, eq, reference_output.data(), output.data(), left.data(), right.data());
        }
        case TconT::ScalarType::Float32: {
            return verify_tccg_case_float(
                case_index,
                eq,
                static_cast<const float*>(output_device),
                static_cast<const float*>(input_left),
                static_cast<const float*>(input_right));
        }
        default:
            throw std::invalid_argument("Unsupported scalar type for TCCG verification");
    }
}
