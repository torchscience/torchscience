#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(SphericalHankelH1Function, spherical_hankel_h_1, n, x)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(spherical_hankel_h_1)

} // namespace torchscience::autograd::special_functions
