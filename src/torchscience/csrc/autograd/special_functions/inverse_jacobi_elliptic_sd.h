#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(InverseJacobiEllipticSdFunction, inverse_jacobi_elliptic_sd, x, k)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(inverse_jacobi_elliptic_sd)

} // namespace torchscience::autograd::special_functions
