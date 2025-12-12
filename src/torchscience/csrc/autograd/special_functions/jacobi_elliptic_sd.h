#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(JacobiEllipticSdFunction, jacobi_elliptic_sd, u, k)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(jacobi_elliptic_sd)

} // namespace torchscience::autograd::special_functions
