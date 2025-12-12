#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(CarlsonEllipticRC, carlson_elliptic_r_c, x, y)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(carlson_elliptic_r_c)

} // namespace torchscience::autograd::special_functions
