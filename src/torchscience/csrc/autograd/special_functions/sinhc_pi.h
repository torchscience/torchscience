#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(SinhcPi, sinhc_pi)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(sinhc_pi)

} // namespace torchscience::autograd::special_functions
