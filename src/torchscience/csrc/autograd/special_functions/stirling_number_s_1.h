#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(StirlingNumberS1, stirling_number_s_1, n, k)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(stirling_number_s_1)

} // namespace torchscience::autograd::special_functions
