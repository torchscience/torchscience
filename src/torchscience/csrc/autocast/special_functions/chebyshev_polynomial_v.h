#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(chebyshev_polynomial_v, n, x)
TORCHSCIENCE_BINARY_AUTOCAST_IMPL(chebyshev_polynomial_v)

} // namespace torchscience::autocast::special_functions
