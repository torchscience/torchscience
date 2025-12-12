#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(spherical_bessel_y, n, x)
TORCHSCIENCE_BINARY_AUTOCAST_IMPL(spherical_bessel_y)

} // namespace torchscience::autocast::special_functions
