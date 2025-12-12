#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(CarlsonEllipticIntegralRC, carlson_elliptic_integral_r_c, x, y)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(carlson_elliptic_integral_r_c)

} // namespace torchscience::autograd::special_functions
