#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_TERNARY_AUTOCAST(associated_legendre_p, n, m, x)
TORCHSCIENCE_TERNARY_AUTOCAST_IMPL(associated_legendre_p)

} // namespace torchscience::autocast::special_functions
