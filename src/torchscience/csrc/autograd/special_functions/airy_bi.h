#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(AiryBi, airy_bi)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(airy_bi)

} // namespace torchscience::autograd::special_functions
