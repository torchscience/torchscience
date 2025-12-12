#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(HankelH1Function, hankel_h_1, nu, x)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(hankel_h_1)

} // namespace torchscience::autograd::special_functions
