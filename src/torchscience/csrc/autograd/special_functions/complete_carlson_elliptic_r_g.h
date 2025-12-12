#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(CompleteCarlsonEllipticRG, complete_carlson_elliptic_r_g, x, y)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(complete_carlson_elliptic_r_g)

} // namespace torchscience::autograd::special_functions
