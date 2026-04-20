/**
 * @file tccg_bench.cpp
 * @author Jinsung Kim (kimjsung@cau.ac.kr)
 * @brief 
 * @version 0.1
 * @date 2026-02-09
 * 
 * @copyright Copyright (c) 2026
 * 
 */
#pragma once
#include "tcont.hpp"
#include <cmath>
#include <string>
#include <cstring>
#include <iostream>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif


/**
 *  TCCG Benchmark TCs
 */
std::vector<TconT::TCEquation> list_tccg_bench = 
{
  // 00: C[a,b,c] -= A[c,d] * B[d,a,b] (dummy)
  { 
    {'a','b','c'}, {'c','d'}, {'d','a','b'}, '-', 
    { {'a',312}, {'b',296}, {'c',296}, {'d',312} } 
  },
  //////////////////////////////////////////////////////////////////////////////
  // 01: C[a,b,c] -= A[b,d,a] * B[d,c]
  { 
    {'a','b','c'}, {'b','d','a'}, {'d','c'}, '-', 
    { {'a',312}, {'b',312}, {'c',24}, {'d',312} } 
  },
  // 02: C[a,b,c] -= A[d,c,a] * B[b,d]
  { 
    {'a','b','c'}, {'d','c','a'}, {'b','d'}, '-', 
    { {'a',312}, {'b',24}, {'c',296}, {'d',312} } 
  },
  // 03: C[a,b,c,d] -= A[d,b,e,a] * B[e,c]
  { 
    {'a','b','c','d'}, {'d','b','e','a'}, {'e','c'}, '-', 
    { {'a',72}, {'b',72}, {'c',24}, {'d',72}, {'e',72} } 
  },
  // 04: C[a,b,c,d] -= A[d,e,c,a] * B[b,e]
  { 
    {'a','b','c','d'}, {'d','e','c','a'}, {'b','e'}, '-', 
    { {'a',72}, {'b',24}, {'c',72}, {'d',72}, {'e',72} } 
  },
  // 05: C[a,b,c,d] -= A[e,b,a,d] * B[c,e]
  { 
    {'a','b','c','d'}, {'e','b','a','d'}, {'c','e'}, '-', 
    { {'a',72}, {'b',72}, {'c',24}, {'d',72}, {'e',72} } 
  },
  // 06: C[a,b,c,d,e] -= A[e,f,b,a,d] * B[c,f]
  { 
    {'a','b','c','d','e'}, {'e','f','b','a','d'}, {'c','f'}, '-', 
    { {'a',48}, {'b',32}, {'c',24}, {'d',32}, {'e',48}, {'f',32} } 
  },
  // 07: C[a,b,c,d,e] -= A[e,c,b,f,a] * B[f,d]
  { 
    {'a','b','c','d','e'}, {'e','c','b','f','a'}, {'f','d'}, '-', 
    { {'a',48}, {'b',32}, {'c',32}, {'d',24}, {'e',48}, {'f',48} } 
  },
  // 08: C[a,b,c,d,e] -= A[e,f,c,a,d] * B[b,f]
  { 
    {'a','b','c','d','e'}, {'e','f','c','a','d'}, {'b','f'}, '-', 
    { {'a',48}, {'b',24}, {'c',32}, {'d',32}, {'e',48}, {'f',32} } 
  },
  //////////////////////////////////////////////////////////////////////////////
  // 09: C[a,b,c,d] -= A[e,a] * B[e,b,c,d]
  { 
    {'a','b','c','d'}, {'e','a'}, {'e','b','c','d'}, '-', 
    { {'a',72}, {'b',72}, {'c',72}, {'d',72}, {'e',72} } 
  },
  // 10: C[a,b,c,d] -= A[e,b] * B[a,e,c,d]
  { 
    {'a','b','c','d'}, {'e','b'}, {'a','e','c','d'}, '-', 
    { {'a',72}, {'b',72}, {'c',72}, {'d',72}, {'e',72} } 
  },
  // 11: C[a,b,c,d] -= A[e,c] * B[a,b,e,d]
  { 
    {'a','b','c','d'}, {'e','c'}, {'a','b','e','d'}, '-', 
    { {'a',72}, {'b',72}, {'c',72}, {'d',72}, {'e',72} } 
  },
  //////////////////////////////////////////////////////////////////////////////
  // 12: C[a,b] -= A[a,c] * B[c,b]
  { 
    {'a','b'}, {'a','c'}, {'c','b'}, '-', 
    { {'a',5136}, {'b',5120}, {'c',5136} } 
  },
  // 13: C[a,b] -= A[a,c,d] * B[d,b,c]
  { 
    {'a','b'}, {'a','c','d'}, {'d','b','c'}, '-', 
    { {'a',312}, {'b',296}, {'c',296}, {'d',312} } 
  },
  // 14: C[a,b] -= A[c,a,d] * B[d,c,b]
  { 
    {'a','b'}, {'c','a','d'}, {'d','c','b'}, '-', 
    { {'a',312}, {'b',296}, {'c',312}, {'d',312} } 
  },
  // 15: C[a,b,c] -= A[a,c,d] * B[d,b]
  { 
    {'a','b','c'}, {'a','c','d'}, {'d','b'}, '-', 
    { {'a',312}, {'b',296}, {'c',296}, {'d',312} } 
  },
  // 16: C[a,b,c] -= A[a,d] * B[b,d,c]
  { 
    {'a','b','c'}, {'a','d'}, {'b','d','c'}, '-', 
    { {'a',312}, {'b',312}, {'c',296}, {'d',296} } 
  },
  // 17: C[a,b,c] -= A[a,d,c] * B[b,d]
  { 
    {'a','b','c'}, {'a','d','c'}, {'b','d'}, '-', 
    { {'a',312}, {'b',312}, {'c',296}, {'d',296} } 
  },
  // 18: C[a,b,c] -= A[a,d,c] * B[d,b]
  { 
    {'a','b','c'}, {'a','d','c'}, {'d','b'}, '-', 
    { {'a',312}, {'b',296}, {'c',296}, {'d',312} } 
  },
  // 19: C[a,b,c] -= A[a,d,e,c] * B[e,b,d]
  { 
    {'a','b','c'}, {'a','d','e','c'}, {'e','b','d'}, '-', 
    { {'a',72}, {'b',72}, {'c',72}, {'d',72}, {'e',72} } 
  },
  //////////////////////////////////////////////////////////////////////////////
  // 20: C[a,b,c,d] -= A[a,e,b,f] * B[d,f,c,e]
  { 
    {'a','b','c','d'}, {'a','e','b','f'}, {'d','f','c','e'}, '-', 
    { {'a',72}, {'b',72}, {'c',72}, {'d',72}, {'e',72}, {'f',72} } 
  },
  // 21: C[a,b,c,d] -= A[a,e,b,f] * B[f,d,e,c]
  { 
    {'a','b','c','d'}, {'a','e','b','f'}, {'f','d','e','c'}, '-', 
    { {'a',72}, {'b',72}, {'c',72}, {'d',72}, {'e',72}, {'f',72} } 
  },
  // 22: C[a,b,c,d] -= A[a,e,c,f] * B[b,f,d,e]
  { 
    {'a','b','c','d'}, {'a','e','c','f'}, {'b','f','d','e'}, '-', 
    { {'a',72}, {'b',72}, {'c',72}, {'d',72}, {'e',72}, {'f',72} } 
  },
  // 23: C[a,b,c,d] -= A[a,e,c,f] * B[f,b,e,d]
  { 
    {'a','b','c','d'}, {'a','e','c','f'}, {'f','b','e','d'}, '-', 
    { {'a',72}, {'b',72}, {'c',72}, {'d',72}, {'e',72}, {'f',72} } 
  },
  // 24: C[a,b,c,d] -= A[a,e,d,f] * B[b,f,c,e]
  { 
    {'a','b','c','d'}, {'a','e','d','f'}, {'b','f','c','e'}, '-', 
    { {'a',72}, {'b',72}, {'c',72}, {'d',72}, {'e',72}, {'f',72} } 
  },
  // 25: C[a,b,c,d] -= A[a,e,d,f] * B[f,b,e,c]
  { 
    {'a','b','c','d'}, {'a','e','d','f'}, {'f','b','e','c'}, '-', 
    { {'a',72}, {'b',72}, {'c',72}, {'d',72}, {'e',72}, {'f',72} } 
  },
  // 26: C[a,b,c,d] -= A[a,e,f,b] * B[f,d,c,e]
  { 
    {'a','b','c','d'}, {'a','e','f','b'}, {'f','d','c','e'}, '-', 
    { {'a',72}, {'b',72}, {'c',72}, {'d',72}, {'e',72}, {'f',72} } 
  },
  // 27: C[a,b,c,d] -= A[a,e,f,c] * B[f,b,e,d]
  { 
    {'a','b','c','d'}, {'a','e','f','c'}, {'f','b','e','d'}, '-', 
    { {'a',72}, {'b',72}, {'c',72}, {'d',72}, {'e',72}, {'f',72} } 
  },
  // 28: C[a,b,c,d] -= A[e,a,f,b] * B[f,d,e,c]
  { 
    {'a','b','c','d'}, {'e','a','f','b'}, {'f','d','e','c'}, '-', 
    { {'a',72}, {'b',72}, {'c',72}, {'d',72}, {'e',72}, {'f',72} } 
  },
  // 29: C[a,b,c,d] -= A[e,a,f,c] * B[b,f,d,e]
  { 
    {'a','b','c','d'}, {'e','a','f','c'}, {'b','f','d','e'}, '-', 
    { {'a',72}, {'b',72}, {'c',72}, {'d',72}, {'e',72}, {'f',72} } 
  },
  // 30: C[a,b,c,d] -= A[e,a,f,d] * B[f,b,e,c]
  { 
    {'a','b','c','d'}, {'e','a','f','d'}, {'f','b','e','c'}, '-', 
    { {'a',72}, {'b',72}, {'c',72}, {'d',72}, {'e',72}, {'f',72} } 
  },
  //////////////////////////////////////////////////////////////////////////////
  // 31: C[a,b,c,d,e,f] -= A[d,e,g,a] * B[g,f,b,c]
  { 
    {'a','b','c','d','e','f'}, {'d','e','g','a'}, {'g','f','b','c'}, '-', 
    { {'a',24}, {'b',16}, {'c',16}, {'d',24}, {'e',16}, {'f',16}, {'g',24} } 
  },
  // 32: C[a,b,c,d,e,f] -= A[d,e,g,b] * B[g,f,a,c]
  { 
    {'a','b','c','d','e','f'}, {'d','e','g','b'}, {'g','f','a','c'}, '-', 
    { {'a',24}, {'b',16}, {'c',16}, {'d',24}, {'e',16}, {'f',16}, {'g',24} } 
  },
  // 33: C[a,b,c,d,e,f] -= A[d,e,g,c] * B[g,f,a,b]
  { 
    {'a','b','c','d','e','f'}, {'d','e','g','c'}, {'g','f','a','b'}, '-', 
    { {'a',24}, {'b',16}, {'c',16}, {'d',24}, {'e',16}, {'f',16}, {'g',24} } 
  },
  // 34: C[a,b,c,d,e,f] -= A[d,f,g,a] * B[g,e,b,c]
  { 
    {'a','b','c','d','e','f'}, {'d','f','g','a'}, {'g','e','b','c'}, '-', 
    { {'a',24}, {'b',16}, {'c',16}, {'d',24}, {'e',16}, {'f',16}, {'g',24} } 
  },
  // 35: C[a,b,c,d,e,f] -= A[d,f,g,b] * B[g,e,a,c]
  { 
    {'a','b','c','d','e','f'}, {'d','f','g','b'}, {'g','e','a','c'}, '-', 
    { {'a',24}, {'b',16}, {'c',16}, {'d',24}, {'e',16}, {'f',16}, {'g',24} } 
  },
  // 36: C[a,b,c,d,e,f] -= A[d,f,g,c] * B[g,e,a,b]
  { 
    {'a','b','c','d','e','f'}, {'d','f','g','c'}, {'g','e','a','b'}, '-', 
    { {'a',24}, {'b',16}, {'c',16}, {'d',24}, {'e',16}, {'f',16}, {'g',24} } 
  },
  // 37: C[a,b,c,d,e,f] -= A[e,f,g,a] * B[g,d,b,c]
  { 
    {'a','b','c','d','e','f'}, {'e','f','g','a'}, {'g','d','b','c'}, '-', 
    { {'a',24}, {'b',16}, {'c',16}, {'d',16}, {'e',24}, {'f',16}, {'g',24} } 
  },
  // 38: C[a,b,c,d,e,f] -= A[e,f,g,b] * B[g,d,a,c]
  { 
    {'a','b','c','d','e','f'}, {'e','f','g','b'}, {'g','d','a','c'}, '-', 
    { {'a',24}, {'b',16}, {'c',16}, {'d',16}, {'e',24}, {'f',16}, {'g',24} } 
  },
  // 39: C[a,b,c,d,e,f] -= A[e,f,g,c] * B[g,d,a,b]
  { 
    {'a','b','c','d','e','f'}, {'e','f','g','c'}, {'g','d','a','b'}, '-', 
    { {'a',24}, {'b',16}, {'c',16}, {'d',16}, {'e',24}, {'f',16}, {'g',24} } 
  },
  // 40: C[a,b,c,d,e,f] -= A[g,d,a,b] * B[e,f,g,c]
  { 
    {'a','b','c','d','e','f'}, {'g','d','a','b'}, {'e','f','g','c'}, '-', 
    { {'a',24}, {'b',16}, {'c',16}, {'d',16}, {'e',24}, {'f',16}, {'g',24} } 
  },
  // 41: C[a,b,c,d,e,f] -= A[g,d,a,c] * B[e,f,g,b]
  { 
    {'a','b','c','d','e','f'}, {'g','d','a','c'}, {'e','f','g','b'}, '-', 
    { {'a',24}, {'b',16}, {'c',16}, {'d',16}, {'e',24}, {'f',16}, {'g',24} } 
  },
  // 42: C[a,b,c,d,e,f] -= A[g,d,b,c] * B[e,f,g,a]
  { 
    {'a','b','c','d','e','f'}, {'g','d','b','c'}, {'e','f','g','a'}, '-', 
    { {'a',24}, {'b',16}, {'c',16}, {'d',16}, {'e',24}, {'f',16}, {'g',24} } 
  },
  // 43: C[a,b,c,d,e,f] -= A[g,e,a,b] * B[d,f,g,c]
  { 
    {'a','b','c','d','e','f'}, {'g','e','a','b'}, {'d','f','g','c'}, '-', 
    { {'a',24}, {'b',16}, {'c',16}, {'d',24}, {'e',16}, {'f',16}, {'g',24} } 
  },
  // 44: C[a,b,c,d,e,f] -= A[g,e,a,c] * B[d,f,g,b]
  { 
    {'a','b','c','d','e','f'}, {'g','e','a','c'}, {'d','f','g','b'}, '-', 
    { {'a',24}, {'b',16}, {'c',16}, {'d',24}, {'e',16}, {'f',16}, {'g',24} } 
  },
  // 45: C[a,b,c,d,e,f] -= A[g,e,b,c] * B[d,f,g,a]
  { 
    {'a','b','c','d','e','f'}, {'g','e','b','c'}, {'d','f','g','a'}, '-', 
    { {'a',24}, {'b',16}, {'c',16}, {'d',24}, {'e',16}, {'f',16}, {'g',24} } 
  },
  // 46: C[a,b,c,d,e,f] -= A[g,f,a,b] * B[d,e,g,c]
  { 
    {'a','b','c','d','e','f'}, {'g','f','a','b'}, {'d','e','g','c'}, '-', 
    { {'a',24}, {'b',16}, {'c',16}, {'d',24}, {'e',16}, {'f',16}, {'g',24} } 
  },
  // 47: C[a,b,c,d,e,f] -= A[g,f,a,c] * B[d,e,g,b]
  { 
    {'a','b','c','d','e','f'}, {'g','f','a','c'}, {'d','e','g','b'}, '-', 
    { {'a',24}, {'b',16}, {'c',16}, {'d',24}, {'e',16}, {'f',16}, {'g',24} } 
  },
  // 48: C[a,b,c,d,e,f] -= A[g,f,b,c] * B[d,e,g,a]
  { 
    {'a','b','c','d','e','f'}, {'g','f','b','c'}, {'d','e','g','a'}, '-', 
    { {'a',24}, {'b',16}, {'c',16}, {'d',24}, {'e',16}, {'f',16}, {'g',24} } 
  },
};

static void print_indices(const std::vector<char>& modes) 
{
  for (auto it = modes.begin(); it != modes.end(); ++it) {
    std::cout << *it;
    if (std::next(it) != modes.end()) {
      std::cout << ",";
    }
  }
}

// 
void print_equation(const TconT::TCEquation& eq)
{
  // Print equation
    std::cout << "[TTGT-cuTT] Equation: C[";
    print_indices(eq.modeC);
    std::cout << "] ";

    if (eq.op == '-')
        std::cout << "-= ";
    else if (eq.op == '+')
        std::cout << "+= ";
    else
        std::cout << eq.op << "= ";

    std::cout << "A[";
    print_indices(eq.modeA);
    std::cout << "] * B[";
    print_indices(eq.modeB);
    std::cout << "]\n";

    // Print extents (sorted for readability)
    std::vector<char> keys;
    for (const auto& kv : eq.extent)
        keys.push_back(kv.first);

    std::sort(keys.begin(), keys.end());

    std::cout << "[TTGT-cuTT] Extents: ";
    for (size_t i = 0; i < keys.size(); ++i)
    {
        char idx = keys[i];
        std::cout << idx << "=" << eq.extent.at(idx);
        if (i != keys.size() - 1)
            std::cout << ", ";
    }
    std::cout << "\n";
}

// 
int64_t compute_tensor_size(const std::vector<char>& modes, const std::unordered_map<char, int64_t>& extent)
{
    int64_t size = 1;
    for (char mode : modes) {
        auto it = extent.find(mode);
        if (it == extent.end()) {
            throw std::runtime_error(std::string("Extent for index '") + mode + "' not found.");
        }
        size *= it->second;
    }
    return size;
}

// Helper to fetch extents from TconT::TCEquation
static inline int extent_of(const TconT::TCEquation& eq, char idx) 
{
    auto it = eq.extent.find(idx);
    if (it == eq.extent.end()) {
        throw std::runtime_error(std::string("Extent for index '") + idx + "' not found.");
    }
    // Your extents are small enough to fit into int for loop bounds
    return static_cast<int>(it->second);
}

//
// "3,a,b,c,-=,1,d,2,c,d,3,d,a,b",             // 00
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
void check_correctness_comparison(int total_size, double* output_host, double* output_device)
{
    if (total_size < 0) {
        fprintf(stderr, "Invalid total size: %d\n", total_size);
        return;
    }
    if (!output_host || !output_device) {
        fprintf(stderr, "Null pointer detected for output arrays.\n");
        return;
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
}