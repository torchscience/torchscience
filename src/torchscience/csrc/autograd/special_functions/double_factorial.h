#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(DoubleFactorial, double_factorial)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(double_factorial)

} // namespace torchscience::autograd::special_functions
