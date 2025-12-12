#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(Digamma, digamma)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(digamma)

} // namespace torchscience::autograd::special_functions
