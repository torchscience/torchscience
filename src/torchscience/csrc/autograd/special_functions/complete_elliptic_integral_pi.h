#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(CompleteEllipticIntegralPi, complete_elliptic_integral_pi, n, k)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(complete_elliptic_integral_pi)

} // namespace torchscience::autograd::special_functions
