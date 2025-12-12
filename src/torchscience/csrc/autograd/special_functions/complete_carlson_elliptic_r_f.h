#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(CompleteCarlsonEllipticRF, complete_carlson_elliptic_r_f, x, y)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(complete_carlson_elliptic_r_f)

} // namespace torchscience::autograd::special_functions
