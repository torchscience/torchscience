#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(NevilleThetaS, neville_theta_s, k, u)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(neville_theta_s)

} // namespace torchscience::autograd::special_functions
