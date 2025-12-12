#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(ChebyshevPolynomialW, chebyshev_polynomial_w, n, x)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(chebyshev_polynomial_w)

} // namespace torchscience::autograd::special_functions
