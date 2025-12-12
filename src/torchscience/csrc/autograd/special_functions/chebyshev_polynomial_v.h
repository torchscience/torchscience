#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(ChebyshevPolynomialV, chebyshev_polynomial_v, n, x)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(chebyshev_polynomial_v)

} // namespace torchscience::autograd::special_functions
