#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(EulerNumberE, euler_number_e)
TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(euler_number_e)

} // namespace torchscience::autograd::special_functions
