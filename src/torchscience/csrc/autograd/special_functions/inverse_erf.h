#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(InverseErf, inverse_erf)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(inverse_erf)

} // namespace torchscience::autograd::special_functions
