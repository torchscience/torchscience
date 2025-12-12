#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_TERNARY_AUTOGRAD(CarlsonEllipticIntegralRD, carlson_elliptic_integral_r_d, x, y, z)
TORCHSCIENCE_TERNARY_AUTOGRAD_IMPL(carlson_elliptic_integral_r_d)

} // namespace torchscience::autograd::special_functions
