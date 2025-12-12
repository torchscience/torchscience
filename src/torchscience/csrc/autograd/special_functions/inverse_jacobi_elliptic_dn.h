#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(InverseJacobiEllipticDnFunction, inverse_jacobi_elliptic_dn, x, k)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(inverse_jacobi_elliptic_dn)

} // namespace torchscience::autograd::special_functions
