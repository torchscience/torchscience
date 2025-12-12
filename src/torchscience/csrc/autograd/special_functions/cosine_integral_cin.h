#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(CosineIntegralCin, cosine_integral_cin)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(cosine_integral_cin)

} // namespace torchscience::autograd::special_functions
