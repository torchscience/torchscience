#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(ChebyshevPolynomialT, chebyshev_polynomial_t, n, x)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(chebyshev_polynomial_t)

} // namespace torchscience::autograd::special_functions
