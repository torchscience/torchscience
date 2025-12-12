#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD_KERNEL(hermite_polynomial_he, n, x)

} // namespace torchscience::autograd::special_functions
