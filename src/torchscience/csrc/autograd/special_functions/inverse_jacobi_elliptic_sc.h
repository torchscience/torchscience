#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(InverseJacobiEllipticScFunction, inverse_jacobi_elliptic_sc, x, k)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(inverse_jacobi_elliptic_sc)

} // namespace torchscience::autograd::special_functions
