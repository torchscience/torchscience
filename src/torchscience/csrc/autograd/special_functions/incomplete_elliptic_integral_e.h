#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(IncompleteEllipticIntegralE, incomplete_elliptic_integral_e, phi, k)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(incomplete_elliptic_integral_e)

} // namespace torchscience::autograd::special_functions
