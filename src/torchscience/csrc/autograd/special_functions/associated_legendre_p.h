#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_TERNARY_AUTOGRAD(AssociatedLegendreP, associated_legendre_p, n, m, x)
TORCHSCIENCE_TERNARY_AUTOGRAD_IMPL(associated_legendre_p)

} // namespace torchscience::autograd::special_functions
