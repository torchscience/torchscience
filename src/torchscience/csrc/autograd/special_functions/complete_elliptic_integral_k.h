#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(CompleteEllipticIntegralK, complete_elliptic_integral_k)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(complete_elliptic_integral_k)

} // namespace torchscience::autograd::special_functions
