#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(ExponentialIntegralEFunction, exponential_integral_e, n, x)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(exponential_integral_e)

} // namespace torchscience::autograd::special_functions
