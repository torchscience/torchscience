#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(RisingFactorial, rising_factorial, x, n)

TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(rising_factorial)

} // namespace torchscience::autograd::special_functions
