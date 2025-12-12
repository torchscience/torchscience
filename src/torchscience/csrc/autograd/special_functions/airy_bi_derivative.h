#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(AiryBiDerivative, airy_bi_derivative)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(airy_bi_derivative)

} // namespace torchscience::autograd::special_functions
