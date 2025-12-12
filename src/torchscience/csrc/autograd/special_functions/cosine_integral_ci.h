#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(CosineIntegralCi, cosine_integral_ci)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(cosine_integral_ci)

} // namespace torchscience::autograd::special_functions
