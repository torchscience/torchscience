#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(incomplete_elliptic_integral_f, phi, k)
TORCHSCIENCE_BINARY_AUTOCAST_IMPL(incomplete_elliptic_integral_f)

} // namespace torchscience::autocast::special_functions
