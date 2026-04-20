#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "tcont.hpp"
#include "cogent.hpp"

// void plan();


// void execute();


namespace TconT
{
    ContractionResult contract(Backend b, TCEquation& desc) {
        switch(b) {
            case Backend::COGENT :
                std::cout << "Contracting with Cogent backend\n";
                return cogent::run_cogent(desc);
            case Backend::TTGT_CUTT :
                std::cout << "Contracting with TTGT_CUTT backend\n";
                // return ttgt_cutt::run_ttgt_cutt(desc);
                break;
            default :
                std::cerr << "Unsupported backend\n";
                return ContractionResult{0.0f, 0.0, false};
        }
    }

    Backend plan(TCEquation& desc) {
        cogent::KernelConfig cfg = cogent::plan_cogent(desc);

        // ttgt plan
        // ttgt_cutt::plan plan = ttgt_cutt::plan_ttgt_cutt(desc);

        // model
        //TconT::model model_result = TconT::model();

        Backend backend;

        return backend;
    }
}
