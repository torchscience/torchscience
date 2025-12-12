#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(carlson_elliptic_r_c, x, y)
TORCHSCIENCE_BINARY_AUTOCAST_IMPL(carlson_elliptic_r_c)

} // namespace torchscience::autocast::special_functions
