#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(legendre_q, n, x)
TORCHSCIENCE_BINARY_AUTOCAST_IMPL(legendre_q)

} // namespace torchscience::autocast::special_functions
