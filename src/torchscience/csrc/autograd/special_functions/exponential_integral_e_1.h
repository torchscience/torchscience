#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(ExponentialIntegralE1, exponential_integral_e_1)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(exponential_integral_e_1)

} // namespace torchscience::autograd::special_functions
