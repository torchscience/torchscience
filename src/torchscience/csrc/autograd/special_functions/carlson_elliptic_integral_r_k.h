#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_TERNARY_AUTOGRAD(CarlsonEllipticIntegralRK, carlson_elliptic_integral_r_k, x, y, z)
TORCHSCIENCE_TERNARY_AUTOGRAD_IMPL(carlson_elliptic_integral_r_k)

} // namespace torchscience::autograd::special_functions
