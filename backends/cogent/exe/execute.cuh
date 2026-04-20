#pragma once
#include "compile.cuh"
#include <cuda.h>
#include <string>
#include <filesystem>
#include <stdexcept>
#include <iostream>
#include "../helper/cuda_helper.cuh"

#define CEIL(a, b) (((a) + (b) - 1) / (b))

inline float execute(const KernelConfig& cfg, double* dA, double* dB, double* dC, 
    double* hA, double* hB, double* hC, const std::unordered_map<int, int64_t>& extents, int equation) {
    // .cubin 없으면 런타임 컴파일
    if(!std::filesystem::exists(cfg.kernel_bin)) {
        std::string cu_path = cfg.kernel_bin;
        if (cu_path.find("bin/") == 0)
            cu_path = cu_path.substr(4);
        cu_path = "code/" + cu_path;

        cu_path.replace(cu_path.rfind(".cubin"), 6, ".cu");

        if(!std::filesystem::exists(cu_path))
            throw std::runtime_error("Neither .cubin nor .cu found: " + cu_path);

        compile_to_cubin(cu_path, cfg.kernel_bin);
    }

    // cubin 로드
    CUmodule   mod;
    CUresult load_res = cuModuleLoad(&mod, cfg.kernel_bin.c_str());
    if (load_res != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(load_res, &errStr);
        throw std::runtime_error(std::string("cuModuleLoad failed: ") + errStr + " / path: " + cfg.kernel_bin);
    }

    CUfunction func;
    CUresult res = cuModuleGetFunction(&func, mod, cfg.kernel_name.c_str());
    if(res != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(res, &errStr);
        cuModuleUnload(mod);
        throw std::runtime_error("cuModuleGetFunction failed: " + cfg.kernel_name + " / " + errStr);
    }

    const uint shm_size = sizeof(double) * (cfg.smem_x + cfg.smem_y) * cfg.stage;

    int num_thread_blocks = 1;
    for(const auto& idx : cfg.external_index) {
        num_thread_blocks *= CEIL((int)extents.at(idx[0]), cfg.tile_sizes.at(idx));
    }

    dim3 gridDim(num_thread_blocks);
    dim3 blockDim(cfg.block_size);

    std::vector<int> external_sizes;
    external_sizes.reserve(cfg.external_index.size());
    for(const auto& idx : cfg.external_index)
        external_sizes.push_back((int)extents.at(idx[0]));

    std::vector<int> internal_sizes;
    internal_sizes.reserve(cfg.internal_index.size());
    for(const auto& idx : cfg.internal_index)
        internal_sizes.push_back((int)extents.at(idx[0]));

    int size_internal = cfg.size_internal;

    std::vector<int> ceils;
    ceils.reserve(cfg.external_index.size());
    for(const auto& idx : cfg.external_index) {
        ceils.push_back(CEIL((int)extents.at(idx[0]), cfg.tile_sizes.at(idx)));
    }

    int stages = cfg.stage;
    int smem_x = cfg.smem_x;
    int smem_y = cfg.smem_y;

    // 3) kernel 실행
    std::vector<void*> args;
    args.push_back(&dC);
    if(cfg.swap_flag) {                      // "a" in t2
        args.push_back(&dB);                    // dev_v2
        args.push_back(&dA);                    // dev_t2
    }
    else {                                 // "a" in v2
        args.push_back(&dA);                  // dev_t2
        args.push_back(&dB);                  // dev_v2
    }

    for(auto& s : external_sizes)
        args.push_back(&s);
    
    for(auto& s : internal_sizes)
        args.push_back(&s);
    
    for(auto& c : ceils)
        args.push_back(&c);

    args.push_back(&size_internal);
    
    // args.push_back(&stages);

    args.push_back(&smem_y);
    args.push_back(&smem_x);

    /*int arg_idx = 0;

    // 0: dC (double*)
    std::cout << "[" << arg_idx++ << "] dC (ptr)        = " << *static_cast<double**>(args[0]) << std::endl;

    // 1: dB or dA (swap_flag에 따라)
    if(cfg.swap_flag) {
        std::cout << "[" << arg_idx++ << "] dB/dev_v2 (ptr) = " << *static_cast<double**>(args[1]) << std::endl;
        std::cout << "[" << arg_idx++ << "] dA/dev_t2 (ptr) = " << *static_cast<double**>(args[2]) << std::endl;
    } else {
        std::cout << "[" << arg_idx++ << "] dA/dev_t2 (ptr) = " << *static_cast<double**>(args[1]) << std::endl;
        std::cout << "[" << arg_idx++ << "] dB/dev_v2 (ptr) = " << *static_cast<double**>(args[2]) << std::endl;
    }

    // external_sizes
    for(int i = 0; i < (int)external_sizes.size(); i++)
        std::cout << "[" << arg_idx++ << "] external_sizes[" << i << "] = "
                << *static_cast<int*>(args[3 + i]) << std::endl;

    int offset = 3 + (int)external_sizes.size();

    // internal_sizes
    for(int i = 0; i < (int)internal_sizes.size(); i++)
        std::cout << "[" << arg_idx++ << "] internal_sizes[" << i << "] = "
                << *static_cast<int*>(args[offset + i]) << std::endl;

    offset += (int)internal_sizes.size();

    // ceils
    for(int i = 0; i < (int)ceils.size(); i++)
        std::cout << "[" << arg_idx++ << "] ceils[" << i << "]         = "
                << *static_cast<int*>(args[offset + i]) << std::endl;

    offset += (int)ceils.size();

    // 마지막 3개: size_internal, smem_y, smem_x
    std::cout << "[" << arg_idx++ << "] size_internal   = " << *static_cast<int*>(args[offset])     << std::endl;
    std::cout << "[" << arg_idx++ << "] smem_y          = " << *static_cast<int*>(args[offset + 1]) << std::endl;
    std::cout << "[" << arg_idx++ << "] smem_x          = " << *static_cast<int*>(args[offset + 2]) << std::endl;

    // 요약
    std::cout << "\n[Summary]" << std::endl;
    std::cout << "  total args       = " << args.size()          << std::endl;
    std::cout << "  gridDim.x        = " << gridDim.x            << std::endl;
    std::cout << "  blockDim.x       = " << blockDim.x           << std::endl;
    std::cout << "  shm_size (bytes) = " << shm_size             << std::endl;
    std::cout << "  swap_flag        = " << cfg.swap_flag         << std::endl;
    */
    const int warm_up = 0;
    for (int i = 0; i < warm_up; i++) {
        CUresult err = cuLaunchKernel(func, gridDim.x, 1, 1, blockDim.x, 1, 1, shm_size, nullptr, args.data(), nullptr);
        if (err != CUDA_SUCCESS) {
            const char* errStr;
            cuGetErrorString(err, &errStr);
            throw std::runtime_error(std::string("cuLaunchKernel failed: ") + errStr);
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int repeats = 1;
    cudaEventRecord(start);
    for (int i = 0; i < repeats; i++) {
        cuLaunchKernel(func, gridDim.x, 1, 1, blockDim.x, 1, 1, shm_size, nullptr, args.data(), nullptr);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    float avg_ms = ms / repeats;

    return avg_ms;
}