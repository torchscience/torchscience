#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(CompleteEllipticIntegralE, complete_elliptic_integral_e)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(complete_elliptic_integral_e)

} // namespace torchscience::autograd::special_functions
