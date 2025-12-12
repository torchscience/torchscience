#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(LegendreQ, legendre_q, n, x)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(legendre_q)

} // namespace torchscience::autograd::special_functions
