#pragma once

#include <memory>

#include "tcont.hpp"

namespace ttgt_cutt_backend
{
    std::shared_ptr<TconT::PlanImpl> plan_ttgt_cutt(const TconT::TCEquation& desc);
}
