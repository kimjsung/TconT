#pragma once
#include <cstddef>

#include "tcont.hpp"

struct VerificationTolerance {
    double abs_tol;
    double rel_tol;
};

VerificationTolerance verification_tolerance_for(TconT::ScalarType scalar_type);
void verify_tccg_case(
    size_t case_index,
    const TconT::TCEquation& eq,
    const void* output_device,
    const void* input_left,
    const void* input_right);
