#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(SphericalHankelH2Function, spherical_hankel_h_2, n, x)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(spherical_hankel_h_2)

} // namespace torchscience::autograd::special_functions
