#pragma once
#include <iostream>
#include <nlohmann/json.hpp>
#include "tcont.hpp"
#include "cogent_helpers.hpp"

namespace cogent
{
    struct KernelConfig {
        std::string kernel_bin;
        std::string kernel_name;                          // "kernel_7"
        std::vector<std::string> external_index;             // ["a","b","c","d"]
        std::vector<std::string> internal_index;             // ["a","b","c","d"]
        std::unordered_map<std::string, int> tile_sizes;  // {"d":8, "a":8, "b":16, "c1":16, "c2":2}
        int block_size;                                   // 128
        int stage;                                        // 3
        int smem_x;                                       // 192
        int smem_y;                                       // 32
        std::vector<int> warp_shape;                      // [4, 4, 8]
        int size_internal;                             // 4
        bool swap_flag;
    };

    TconT::ContractionResult run_cogent(TconT::TCEquation& desc) {
        std::cout << "[Backend] Cogent contract called\n";

        typedef double TypeA;
        typedef double TypeB;
        typedef double TypeC;

        // Size of each tensors
        size_t elementsA = 1, elementsB = 1, elementsC = 1;
        for(auto mode : desc.modeA) 
            elementsA *= desc.extent[mode];
        for(auto mode : desc.modeB)
            elementsB *= desc.extent[mode];
        for(auto mode : desc.modeC)
            elementsC *= desc.extent[mode];
        
        size_t sizeA = sizeof(TypeA) * elementsA;
        size_t sizeB = sizeof(TypeB) * elementsB;
        size_t sizeC = sizeof(TypeC) * elementsC;

        // Allocate device memory
        double *dA, *dB, *dC;
        HANDLE_CUDA_ERROR(cudaMalloc((void**)&dA, sizeA));
        HANDLE_CUDA_ERROR(cudaMalloc((void**)&dB, sizeB));
        HANDLE_CUDA_ERROR(cudaMalloc((void**)&dC, sizeC));

        // Allocate host memory
        TypeA *hA = (TypeA*)malloc(sizeof(TypeA) * elementsA);
        TypeB *hB = (TypeB*)malloc(sizeof(TypeB) * elementsB);
        TypeC *hC = (TypeC*)calloc(sizeof(TypeC), elementsC);
        TypeC *hC_chk = (TypeC*)calloc(elementsC, sizeof(TypeC));
        
        // Initialize data
        pre_Initializing_Input_Tensors(hA, elementsA, hB, elementsB);

        // Copy to device
        HANDLE_CUDA_ERROR(cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice));
        HANDLE_CUDA_ERROR(cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice));
        HANDLE_CUDA_ERROR(cudaMemset(dC, 0.0, sizeC));

        return TconT::ContractionResult{0.0f, 0.0, true};
    };

    cogent::KernelConfig plan_cogent(TconT::TCEquation& desc) {
        std::vector<int64_t> extentC;
        std::vector<int64_t> extentA;
        std::vector<int64_t> extentB;
        
        for(auto mode : desc.modeC) {
            extentC.push_back(desc.extent[mode]);
        }
        for(auto mode : desc.modeA) {
            extentA.push_back(desc.extent[mode]);
        }
        for(auto mode : desc.modeB) {
            extentB.push_back(desc.extent[mode]);
        }
        
        nlohmann::json input;
        input["modeC"] = desc.modeC;
        input["extentC"] = extentC;
        input["modeA"] = desc.modeA;
        input["extentA"] = extentA;
        input["modeB"] = desc.modeB;
        input["extentB"] = extentB;
        input["op"] = desc.op;

        std::string cmd = "python3 plan/tc_plan.py '" + input.dump() + "'";

        FILE* pipe = popen(cmd.c_str(), "r");
        char buf[4096];
        std::string result;
        
        while (fgets(buf, sizeof(buf), pipe))
            result += buf;
        
        pclose(pipe);

        nlohmann::json output = nlohmann::json::parse(result);

        cogent::KernelConfig cfg;
        std::string kernel_bin = output["kernel_bin"].get<std::string>();

        cfg.kernel_bin  = "bin/" + kernel_bin + ".cubin";

        cfg.kernel_name   = output["kernel_name"].get<std::string>();
        cfg.external_index = output["external_index"].get<std::vector<std::string>>();
        cfg.internal_index = output["internal_index"].get<std::vector<std::string>>();
        cfg.block_size    = output["block_size"].get<int>();
        cfg.stage         = output["stage"][0].get<int>();
        cfg.smem_x        = output["smem_x"].get<int>();
        cfg.smem_y        = output["smem_y"].get<int>();
        cfg.warp_shape     = output["warp_shape"].get<std::vector<int>>();
        cfg.size_internal = output["internal"].get<int>();
        cfg.swap_flag     = output["swap_flag"].get<bool>();

        // tile_sizes: [["d",8],["a",8],...] → unordered_map
        for (const auto& pair : output["tile_sizes"])
            cfg.tile_sizes[pair[0].get<std::string>()] = pair[1].get<int>();
            
        return cfg;
    };
}
