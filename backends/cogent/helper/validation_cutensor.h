#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <assert.h>
#include <cutensor.h>
#include <cuda_runtime.h>
#include "cuda_helper.cuh"


bool post_Correctness_cutensor(double* C_result, double* C_h, double* C_d, double* A_d, double* B_d, std::vector<int> modeA, std::vector<int> modeB, std::vector<int> modeC, std::vector<int64_t> extentA, std::vector<int64_t> extentB, std::vector<int64_t> extentC, size_t sizeC, size_t elementsC) {
    // Host element type definition
    typedef double TypeA;
    typedef double TypeB;
    typedef double TypeC;
    typedef double TypeCompute;

    // CUDA types
    cutensorDataType_t typeA = CUTENSOR_R_64F;
    cutensorDataType_t typeB = CUTENSOR_R_64F;
    cutensorDataType_t typeC = CUTENSOR_R_64F;
    cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_64F;
    
    // Alignment of the global-memory device pointers (bytes)
    const uint32_t kAlignment = 128;
    assert(uintptr_t(A_d) % kAlignment == 0);
    assert(uintptr_t(B_d) % kAlignment == 0);
    assert(uintptr_t(C_d) % kAlignment == 0);

    cutensorHandle_t handle;
    HANDLE_ERROR(cutensorCreate(&handle));

    cutensorTensorDescriptor_t descA;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                &descA,
                modeA.size(),
                extentA.data(),
                NULL,/*stride*/
                typeA, kAlignment));
    cutensorTensorDescriptor_t descB;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                &descB,
                modeB.size(),
                extentB.data(),
                NULL,/*stride*/
                typeB, kAlignment));
    cutensorTensorDescriptor_t descC;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                &descC,
                modeC.size(),
                extentC.data(),
                NULL,/*stride*/
                typeC, kAlignment));

    cutensorOperationDescriptor_t desc;
    HANDLE_ERROR(cutensorCreateContraction(handle, 
                &desc,
                descA, modeA.data(), /* unary operator A*/CUTENSOR_OP_IDENTITY,
                descB, modeB.data(), /* unary operator B*/CUTENSOR_OP_IDENTITY,
                descC, modeC.data(), /* unary operator C*/CUTENSOR_OP_IDENTITY,
                descC, modeC.data(),
                descCompute));

    cutensorDataType_t scalarType;
    HANDLE_ERROR(cutensorOperationDescriptorGetAttribute(handle,
                desc,
                CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                (void*)&scalarType,
                sizeof(scalarType)));
    assert(scalarType == CUTENSOR_R_64F);

    TypeCompute alpha = (TypeCompute)1.0;
    TypeCompute beta  = (TypeCompute)0.0;

    const cutensorAlgo_t algo = CUTENSOR_ALGO_GETT;

    cutensorPlanPreference_t planPref;
    HANDLE_ERROR(cutensorCreatePlanPreference(
                handle,
                &planPref,
                algo,
                CUTENSOR_JIT_MODE_NONE));

    uint64_t workspaceSizeEstimate = 0;
    const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
    HANDLE_ERROR(cutensorEstimateWorkspaceSize(handle,
                desc,
                planPref,
                workspacePref,
                &workspaceSizeEstimate));

    cutensorPlan_t plan;
    HANDLE_ERROR(cutensorCreatePlan(handle,
                &plan,
                desc,
                planPref,
                workspaceSizeEstimate));

    uint64_t actualWorkspaceSize = 0;
    HANDLE_ERROR(cutensorPlanGetAttribute(handle,
                plan,
                CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                &actualWorkspaceSize,
                sizeof(actualWorkspaceSize)));
    
    assert(actualWorkspaceSize <= workspaceSizeEstimate);
    
    void *work = nullptr;
    if (actualWorkspaceSize > 0)
    {
        CHECK_CUDA(cudaMalloc(&work, actualWorkspaceSize));
        assert(uintptr_t(work) % 128 == 0); // workspace must be aligned to 128 byte-boundary
    }

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    HANDLE_ERROR(cutensorContract(handle,
                    plan,
                    (void*) &alpha, A_d, B_d,
                    (void*) &beta,  C_d, C_d, 
                    work, actualWorkspaceSize, stream));
    CHECK_CUDA(cudaDeviceSynchronize());

    std::cout << "Contraction completed" << std::endl;
    
    CHECK_CUDA(cudaMemcpy(C_h, C_d, sizeC, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());

    printf ("======================================= Correctness Check ==========================================\n");
    double   epsilon = 0.00000001;
    size_t   diff    = 0;
    size_t   same    = 0;
    size_t 	 i;

	#pragma omp parallel for reduction(+:diff, same)
    for (i = 0; i < elementsC; i++)
    {
        double check = C_h[i] - C_result[i];
        if (check < 0) check *= -1;
        if (check > epsilon)
        {
            diff++;
            // if (diff < 8)
            //     printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, C[i], device[i], check);
        }
        else
        {
            same++;
        }
    }

    printf (" >>> PASSED: %10ld among %10ld in t3\n", same, elementsC);
    printf (" >>> ERROR : %10ld among %10ld in t3\n", diff, elementsC);
    // printf (" >>> Total Operations: %'lld\n", ops * 2);
    printf ("====================================================================================================\n");

	bool flag;
	if(diff == 0)
		flag = true;
	else
		flag = false;

	return flag;
}