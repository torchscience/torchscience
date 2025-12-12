#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_TERNARY_AUTOGRAD(CarlsonEllipticIntegralRM, carlson_elliptic_integral_r_m, x, y, z)
TORCHSCIENCE_TERNARY_AUTOGRAD_IMPL(carlson_elliptic_integral_r_m)

} // namespace torchscience::autograd::special_functions
