#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST_KERNEL(shifted_chebyshev_polynomial_u, n, x)

} // namespace torchscience::autocast::special_functions
