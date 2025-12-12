#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(AiryAi, airy_ai)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(airy_ai)

} // namespace torchscience::autograd::special_functions
