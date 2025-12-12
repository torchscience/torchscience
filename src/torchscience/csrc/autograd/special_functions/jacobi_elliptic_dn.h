#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(JacobiEllipticDnFunction, jacobi_elliptic_dn, u, k)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(jacobi_elliptic_dn)

} // namespace torchscience::autograd::special_functions
