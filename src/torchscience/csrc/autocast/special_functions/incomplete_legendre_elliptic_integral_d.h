#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(incomplete_legendre_elliptic_integral_d, phi, k)
TORCHSCIENCE_BINARY_AUTOCAST_IMPL(incomplete_legendre_elliptic_integral_d)

} // namespace torchscience::autocast::special_functions
