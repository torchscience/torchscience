#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(SineIntegralSi, sine_integral_si)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(sine_integral_si)

} // namespace torchscience::autograd::special_functions
