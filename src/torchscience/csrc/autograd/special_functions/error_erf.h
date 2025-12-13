#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(ErrorErf, error_erf)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(error_erf)

} // namespace torchscience::autograd::special_functions
