#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(legendre_p, n, x)
TORCHSCIENCE_BINARY_AUTOCAST_IMPL(legendre_p)

} // namespace torchscience::autocast::special_functions
