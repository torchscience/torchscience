#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(complete_elliptic_integral_pi, n, k)
TORCHSCIENCE_BINARY_AUTOCAST_IMPL(complete_elliptic_integral_pi)

} // namespace torchscience::autocast::special_functions
