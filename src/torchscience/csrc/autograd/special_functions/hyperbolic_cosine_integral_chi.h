#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(HyperbolicCosineIntegralChi, hyperbolic_cosine_integral_chi)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(hyperbolic_cosine_integral_chi)

} // namespace torchscience::autograd::special_functions
