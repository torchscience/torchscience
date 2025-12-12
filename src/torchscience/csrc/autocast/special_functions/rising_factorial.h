#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(rising_factorial, x, n)

TORCHSCIENCE_BINARY_AUTOCAST_IMPL(rising_factorial)

} // namespace torchscience::autocast::special_functions
