#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_TERNARY_AUTOCAST(legendre_elliptic_integral_pi, n, phi, k)
TORCHSCIENCE_TERNARY_AUTOCAST_IMPL(legendre_elliptic_integral_pi)

} // namespace torchscience::autocast::special_functions
