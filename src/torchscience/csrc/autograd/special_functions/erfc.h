#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(Erfc, erfc)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(erfc)

} // namespace torchscience::autograd::special_functions
