#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(LogarithmicIntegralLi, logarithmic_integral_li)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(logarithmic_integral_li)

} // namespace torchscience::autograd::special_functions
