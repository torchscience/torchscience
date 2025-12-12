#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(modified_bessel_i_derivative, nu, x)

TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(modified_bessel_i_derivative)

} // namespace torchscience::autograd::special_functions
