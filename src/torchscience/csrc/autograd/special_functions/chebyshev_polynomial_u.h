#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(ChebyshevPolynomialU, chebyshev_polynomial_u, n, x)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(chebyshev_polynomial_u)

} // namespace torchscience::autograd::special_functions
