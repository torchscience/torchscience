#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(TangentNumberT2, tangent_number_t_2)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(tangent_number_t_2)

} // namespace torchscience::autograd::special_functions
