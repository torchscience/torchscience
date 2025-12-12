#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(complete_carlson_elliptic_r_g, x, y)
TORCHSCIENCE_BINARY_AUTOCAST_IMPL(complete_carlson_elliptic_r_g)

} // namespace torchscience::autocast::special_functions
