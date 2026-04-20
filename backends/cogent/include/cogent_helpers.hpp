#pragma once
#include <stdio.h>
#include <cuda_runtime.h>

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
    if( err != cudaSuccess )                                      \
    {   printf("CUDA Error: %s\n", cudaGetErrorName(err));          \
        printf("Error: %s\n", cudaGetErrorString(err)); exit(-1); } \
};