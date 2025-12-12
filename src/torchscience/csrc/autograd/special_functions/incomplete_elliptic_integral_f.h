#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(IncompleteEllipticIntegralF, incomplete_elliptic_integral_f, phi, k)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(incomplete_elliptic_integral_f)

} // namespace torchscience::autograd::special_functions
