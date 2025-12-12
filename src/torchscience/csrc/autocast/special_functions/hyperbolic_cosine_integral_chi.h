#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(hyperbolic_cosine_integral_chi)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(hyperbolic_cosine_integral_chi)

} // namespace torchscience::autocast::special_functions
