#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>

namespace TconT
{
    enum class Backend { 
        COGENT, 
        TTGT_CUTT
    };
    
    struct TCEquation {
        std::vector<char> modeC;
        std::vector<char> modeA;
        std::vector<char> modeB;
        char op = '?';

        std::unordered_map<char, int64_t> extent;
    };

    struct ContractionResult {
        float avg_ms;
        double gflops;
        bool coorect;
    };

    ContractionResult contract(Backend b, TCEquation& desc);

    Backend plan(TCEquation& desc);
}
