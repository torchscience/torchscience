#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(BinomialCoefficient, binomial_coefficient, n, k)

TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(binomial_coefficient)

} // namespace torchscience::autograd::special_functions
