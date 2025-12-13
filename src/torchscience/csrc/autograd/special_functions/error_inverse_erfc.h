#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(ErrorInverseErfc, error_inverse_erfc)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(error_inverse_erfc)

} // namespace torchscience::autograd::special_functions
