#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(FibonacciNumberF, fibonacci_number_f)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(fibonacci_number_f)

} // namespace torchscience::autograd::special_functions
