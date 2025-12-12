#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(ModifiedBesselKDerivative, modified_bessel_k_derivative, nu, x)

TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(modified_bessel_k_derivative)

} // namespace torchscience::autograd::special_functions
