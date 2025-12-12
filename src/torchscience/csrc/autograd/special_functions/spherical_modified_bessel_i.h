#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(SphericalModifiedBesselIFunction, spherical_modified_bessel_i, n, x)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(spherical_modified_bessel_i)

} // namespace torchscience::autograd::special_functions
