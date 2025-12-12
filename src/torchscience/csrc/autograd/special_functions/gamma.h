#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(Gamma, gamma)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(gamma)

} // namespace torchscience::autograd::special_functions
