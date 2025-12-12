#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(NevilleThetaN, neville_theta_n, k, u)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(neville_theta_n)

} // namespace torchscience::autograd::special_functions
