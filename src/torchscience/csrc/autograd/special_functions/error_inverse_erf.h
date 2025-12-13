#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(ErrorInverseErf, error_inverse_erf)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(error_inverse_erf)

} // namespace torchscience::autograd::special_functions
