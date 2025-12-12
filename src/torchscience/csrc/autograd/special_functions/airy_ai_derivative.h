#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(AiryAiDerivative, airy_ai_derivative)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(airy_ai_derivative)

} // namespace torchscience::autograd::special_functions
