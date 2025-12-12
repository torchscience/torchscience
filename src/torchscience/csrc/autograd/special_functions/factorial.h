#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(Factorial, factorial)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(factorial)

} // namespace torchscience::autograd::special_functions
