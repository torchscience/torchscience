#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(ErrorErfc, error_erfc)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(error_erfc)

} // namespace torchscience::autograd::special_functions
