#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(SincPi, sinc_pi)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(sinc_pi)

} // namespace torchscience::autograd::special_functions
