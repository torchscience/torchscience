#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(JacobiEllipticScFunction, jacobi_elliptic_sc, u, k)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(jacobi_elliptic_sc)

} // namespace torchscience::autograd::special_functions
