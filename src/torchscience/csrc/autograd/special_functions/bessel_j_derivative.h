#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(BesselJDerivative, bessel_j_derivative, nu, x)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(bessel_j_derivative)

} // namespace torchscience::autograd::special_functions
