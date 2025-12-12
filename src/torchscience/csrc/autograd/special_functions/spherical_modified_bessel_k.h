#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(SphericalModifiedBesselKFunction, spherical_modified_bessel_k, n, x)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(spherical_modified_bessel_k)

} // namespace torchscience::autograd::special_functions
