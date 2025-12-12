#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(SinPi, sin_pi)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(sin_pi)

} // namespace torchscience::autograd::special_functions
