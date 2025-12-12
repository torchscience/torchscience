#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(FallingFactorial, falling_factorial, x, n)

TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(falling_factorial)

} // namespace torchscience::autograd::special_functions
