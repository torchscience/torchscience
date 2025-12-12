#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(LegendreP, legendre_p, n, x)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(legendre_p)

} // namespace torchscience::autograd::special_functions
