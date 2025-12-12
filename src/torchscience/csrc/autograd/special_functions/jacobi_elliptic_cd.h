#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(JacobiEllipticCdFunction, jacobi_elliptic_cd, u, k)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(jacobi_elliptic_cd)

} // namespace torchscience::autograd::special_functions
