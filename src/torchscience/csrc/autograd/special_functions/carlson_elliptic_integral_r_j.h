#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_QUATERNARY_AUTOGRAD(CarlsonEllipticIntegralRJ, carlson_elliptic_integral_r_j, x, y, z, p)
TORCHSCIENCE_QUATERNARY_AUTOGRAD_IMPL(carlson_elliptic_integral_r_j)

} // namespace torchscience::autograd::special_functions
