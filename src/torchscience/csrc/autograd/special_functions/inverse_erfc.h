#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(InverseErfc, inverse_erfc)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(inverse_erfc)

} // namespace torchscience::autograd::special_functions
