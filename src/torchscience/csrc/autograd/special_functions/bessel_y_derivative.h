#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(bessel_y_derivative, nu, x)

TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(bessel_y_derivative)

} // namespace torchscience::autograd::special_functions
