#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(IncompleteLegendreEllipticIntegralD, incomplete_legendre_elliptic_integral_d, phi, k)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(incomplete_legendre_elliptic_integral_d)

} // namespace torchscience::autograd::special_functions
