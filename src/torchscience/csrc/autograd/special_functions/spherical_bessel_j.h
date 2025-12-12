#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(SphericalBesselJFunction, spherical_bessel_j, n, x)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(spherical_bessel_j)

} // namespace torchscience::autograd::special_functions
