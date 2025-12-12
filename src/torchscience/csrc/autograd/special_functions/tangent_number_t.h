#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(TangentNumberT, tangent_number_t)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(tangent_number_t)

} // namespace torchscience::autograd::special_functions
