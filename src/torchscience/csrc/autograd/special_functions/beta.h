#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(BetaFunction, beta, a, b)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(beta)

} // namespace torchscience::autograd::special_functions
