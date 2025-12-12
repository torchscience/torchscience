#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(beta, a, b)
TORCHSCIENCE_BINARY_AUTOCAST_IMPL(beta)

} // namespace torchscience::autocast::special_functions
