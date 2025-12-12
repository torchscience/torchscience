#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(NevilleThetaC, neville_theta_c, k, u)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(neville_theta_c)

} // namespace torchscience::autograd::special_functions
