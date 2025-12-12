#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(SphericalBesselYFunction, spherical_bessel_y, n, x)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(spherical_bessel_y)

} // namespace torchscience::autograd::special_functions
