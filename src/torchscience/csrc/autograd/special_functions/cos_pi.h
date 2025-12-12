#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(CosPi, cos_pi)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(cos_pi)

} // namespace torchscience::autograd::special_functions
