#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(NevilleThetaD, neville_theta_d, k, u)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(neville_theta_d)

} // namespace torchscience::autograd::special_functions
