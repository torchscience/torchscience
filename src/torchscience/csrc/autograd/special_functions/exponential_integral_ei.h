#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(ExponentialIntegralEi, exponential_integral_ei)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(exponential_integral_ei)

} // namespace torchscience::autograd::special_functions
