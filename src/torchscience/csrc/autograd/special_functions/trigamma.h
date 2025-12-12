#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(Trigamma, trigamma)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(trigamma)

} // namespace torchscience::autograd::special_functions
