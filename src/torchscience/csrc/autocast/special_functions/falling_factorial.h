#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(falling_factorial, x, n)

TORCHSCIENCE_BINARY_AUTOCAST_IMPL(falling_factorial)

} // namespace torchscience::autocast::special_functions
