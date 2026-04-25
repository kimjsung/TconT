#pragma once
#include <stdio.h>
#include <cuda_runtime.h>

#define HANDLE_CUDA_ERROR(call)														 \
	do {																		 \
		cudaError_t status_ = call;												 \
		if(status_ != cudaSuccess) {											 \
			fprintf(stderr, "CUDA error (%s:%d) : %s:%s\n", __FILE__, __LINE__,  \
					cudaGetErrorName(status_), cudaGetErrorString(status_));	 \
			exit(EXIT_FAILURE);													 \
		}																		 \
	} while(0)


#define HANDLE_ERROR(x)                                                         \
{ const auto err = x;                                                           \
if( err != CUTENSOR_STATUS_SUCCESS )                                            \
    {                                                                           \
        const char* errName = nullptr;                                          \
        switch(err) {                                                           \
            case CUTENSOR_STATUS_SUCCESS:                                       \
                errName = "CUTENSOR_STATUS_SUCCESS"; break;                     \
            case CUTENSOR_STATUS_NOT_INITIALIZED:                               \
                errName = "CUTENSOR_STATUS_NOT_INITIALIZED"; break;             \
            case CUTENSOR_STATUS_ALLOC_FAILED:                                  \
                errName = "CUTENSOR_STATUS_ALLOC_FAILED"; break;                \
            case CUTENSOR_STATUS_INVALID_VALUE:                                 \
                errName = "CUTENSOR_STATUS_INVALID_VALUE"; break;               \
            case CUTENSOR_STATUS_ARCH_MISMATCH:                                 \
                errName = "CUTENSOR_STATUS_ARCH_MISMATCH"; break;               \
            case CUTENSOR_STATUS_MAPPING_ERROR:                                 \
                errName = "CUTENSOR_STATUS_MAPPING_ERROR"; break;               \
            case CUTENSOR_STATUS_EXECUTION_FAILED:                              \
                errName = "CUTENSOR_STATUS_EXECUTION_FAILED"; break;            \
            case CUTENSOR_STATUS_INTERNAL_ERROR:                                \
                errName = "CUTENSOR_STATUS_INTERNAL_ERROR"; break;              \
            case CUTENSOR_STATUS_NOT_SUPPORTED:                                 \
                errName = "CUTENSOR_STATUS_NOT_SUPPORTED"; break;               \
            case CUTENSOR_STATUS_LICENSE_ERROR:                                 \
                errName = "CUTENSOR_STATUS_LICENSE_ERROR"; break;               \
            case CUTENSOR_STATUS_CUBLAS_ERROR:                                  \
                errName = "CUTENSOR_STATUS_CUBLAS_ERROR"; break;                \
            case CUTENSOR_STATUS_CUDA_ERROR:                                    \
                errName = "CUTENSOR_STATUS_CUDA_ERROR"; break;                  \
            case CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE:                        \
                errName = "CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE"; break;      \
            case CUTENSOR_STATUS_INSUFFICIENT_DRIVER:                           \
                errName = "CUTENSOR_STATUS_INSUFFICIENT_DRIVER"; break;         \
            case CUTENSOR_STATUS_IO_ERROR:                                      \
                errName = "CUTENSOR_STATUS_IO_ERROR"; break;                    \
            default:                                                            \
                errName = "CUTENSOR_STATUS_UNKNOWN"; break;                     \
        }                                                                       \
        printf("Error in %s::%s() at line %d:\n"                               \
               "  Status : %s\n"                                                \
               "  Code   : %d\n"                                                \
               "  Message: %s\n",                                               \
               __FILE__, __func__, __LINE__,                                    \
               errName, (int)err, cutensorGetErrorString(err));                 \
        exit(-1);                                                               \
    }                                                                           \
};
