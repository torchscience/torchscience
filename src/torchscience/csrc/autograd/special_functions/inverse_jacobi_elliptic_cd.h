#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(InverseJacobiEllipticCdFunction, inverse_jacobi_elliptic_cd, x, k)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(inverse_jacobi_elliptic_cd)

} // namespace torchscience::autograd::special_functions
