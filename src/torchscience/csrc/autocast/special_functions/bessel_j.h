#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(bessel_j, nu, x)
TORCHSCIENCE_BINARY_AUTOCAST_IMPL(bessel_j)

} // namespace torchscience::autocast::special_functions
