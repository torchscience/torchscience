#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(spherical_modified_bessel_k, n, x)
TORCHSCIENCE_BINARY_AUTOCAST_IMPL(spherical_modified_bessel_k)

} // namespace torchscience::autocast::special_functions
