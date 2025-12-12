#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(stirling_number_s_1, n, k)
TORCHSCIENCE_BINARY_AUTOCAST_IMPL(stirling_number_s_1)

} // namespace torchscience::autocast::special_functions
