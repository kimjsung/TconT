#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda.h>

#include "tcont.hpp"

namespace cogent
{
    struct KernelConfig {
        std::string kernel_bin;
        std::string kernel_name;
        TconT::ScalarType scalar_type = TconT::ScalarType::Float64;
        std::vector<std::string> external_index;
        std::vector<std::string> internal_index;
        std::unordered_map<std::string, int> tile_sizes;
        int block_size = 0;
        int stage = 0;
        int smem_x = 0;
        int smem_y = 0;
        std::vector<int> warp_shape;
        int size_internal = 0;
        bool swap_flag = false;
    };

    struct PreparedKernel {
        KernelConfig config;
        CUmodule module = nullptr;
        CUfunction function = nullptr;
        unsigned int shared_mem_bytes = 0;
        unsigned int grid_dim_x = 0;
        unsigned int block_dim_x = 0;
        std::vector<int> external_sizes;
        std::vector<int> internal_sizes;
        std::vector<int> ceils;

        ~PreparedKernel();

        PreparedKernel() = default;
        PreparedKernel(const PreparedKernel&) = delete;
        PreparedKernel& operator=(const PreparedKernel&) = delete;
        PreparedKernel(PreparedKernel&& other) noexcept;
        PreparedKernel& operator=(PreparedKernel&& other) noexcept;
    };

    std::shared_ptr<TconT::PlanImpl> plan_cogent(const TconT::TCEquation& desc);
}
