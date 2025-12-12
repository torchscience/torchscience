#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(LogGamma, log_gamma)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(log_gamma)

} // namespace torchscience::autograd::special_functions
