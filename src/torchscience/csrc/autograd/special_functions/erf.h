#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(Erf, erf)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(erf)

} // namespace torchscience::autograd::special_functions
