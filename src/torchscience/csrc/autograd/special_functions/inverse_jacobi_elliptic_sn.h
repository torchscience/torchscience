#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(InverseJacobiEllipticSnFunction, inverse_jacobi_elliptic_sn, x, k)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(inverse_jacobi_elliptic_sn)

} // namespace torchscience::autograd::special_functions
