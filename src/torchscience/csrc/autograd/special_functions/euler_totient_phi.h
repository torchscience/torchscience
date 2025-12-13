#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(EulerTotientPhi, euler_totient_phi)
TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(euler_totient_phi)

} // namespace torchscience::autograd::special_functions
