#pragma once
#include <string>

#include "cogent.hpp"
#include "ttgt_cutt.hpp"


namespace TconT {
    enum class Backend { COGENT, TTGT_CUTT };
    
    inline void contract(Backend b) {
        if (b == Backend::COGENT) {
            //
        } else {
            // 
        }
    }
}
