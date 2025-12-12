#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(spherical_modified_bessel_i, n, x)
TORCHSCIENCE_BINARY_AUTOCAST_IMPL(spherical_modified_bessel_i)

} // namespace torchscience::autocast::special_functions
