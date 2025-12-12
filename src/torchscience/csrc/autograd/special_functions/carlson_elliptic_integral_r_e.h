#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_TERNARY_AUTOGRAD(CarlsonEllipticIntegralRE, carlson_elliptic_integral_r_e, x, y, z)
TORCHSCIENCE_TERNARY_AUTOGRAD_IMPL(carlson_elliptic_integral_r_e)

} // namespace torchscience::autograd::special_functions
