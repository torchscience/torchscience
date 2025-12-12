#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(StirlingNumberS2, stirling_number_s_2, n, k)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(stirling_number_s_2)

} // namespace torchscience::autograd::special_functions
