#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(BesselJFunction, bessel_j, nu, x)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(bessel_j)

} // namespace torchscience::autograd::special_functions
