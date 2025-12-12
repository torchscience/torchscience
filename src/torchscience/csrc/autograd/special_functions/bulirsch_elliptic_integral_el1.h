#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(BulirschEllipticIntegralEl1Function, bulirsch_elliptic_integral_el1, x, kc)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(bulirsch_elliptic_integral_el1)

} // namespace torchscience::autograd::special_functions
