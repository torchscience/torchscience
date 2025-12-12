#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(spherical_hankel_h_1, n, x)

TORCHSCIENCE_BINARY_AUTOCAST_IMPL(spherical_hankel_h_1)

} // namespace torchscience::autocast::special_functions
