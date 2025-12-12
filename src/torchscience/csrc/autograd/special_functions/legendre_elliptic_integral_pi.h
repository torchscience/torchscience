#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_TERNARY_AUTOGRAD(LegendreEllipticIntegralPi, legendre_elliptic_integral_pi, n, phi, k)
TORCHSCIENCE_TERNARY_AUTOGRAD_IMPL(legendre_elliptic_integral_pi)

} // namespace torchscience::autograd::special_functions
