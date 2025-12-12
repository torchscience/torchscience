#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(CompleteLegendreEllipticIntegralD, complete_legendre_elliptic_integral_d)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(complete_legendre_elliptic_integral_d)

} // namespace torchscience::autograd::special_functions
