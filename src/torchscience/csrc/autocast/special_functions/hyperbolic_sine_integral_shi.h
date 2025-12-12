#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(hyperbolic_sine_integral_shi)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(hyperbolic_sine_integral_shi)

} // namespace torchscience::autocast::special_functions
