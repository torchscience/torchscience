#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(HyperbolicSineIntegralShi, hyperbolic_sine_integral_shi)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(hyperbolic_sine_integral_shi)

} // namespace torchscience::autograd::special_functions
