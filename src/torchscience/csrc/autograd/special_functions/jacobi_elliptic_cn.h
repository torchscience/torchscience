#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(JacobiEllipticCnFunction, jacobi_elliptic_cn, u, k)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(jacobi_elliptic_cn)

} // namespace torchscience::autograd::special_functions
