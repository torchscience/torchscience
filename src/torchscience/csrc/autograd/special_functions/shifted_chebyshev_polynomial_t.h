#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD_KERNEL(shifted_chebyshev_polynomial_t, n, x)

} // namespace torchscience::autograd::special_functions
