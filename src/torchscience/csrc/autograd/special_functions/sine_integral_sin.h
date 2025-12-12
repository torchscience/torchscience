#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(SineIntegralSin, sine_integral_sin)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(sine_integral_sin)

} // namespace torchscience::autograd::special_functions
