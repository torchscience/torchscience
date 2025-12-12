#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(spherical_bessel_j, n, x)
TORCHSCIENCE_BINARY_AUTOCAST_IMPL(spherical_bessel_j)

} // namespace torchscience::autocast::special_functions
