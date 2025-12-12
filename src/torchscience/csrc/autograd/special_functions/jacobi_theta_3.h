#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(JacobiTheta3, jacobi_theta_3, z, q)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(jacobi_theta_3)

} // namespace torchscience::autograd::special_functions
