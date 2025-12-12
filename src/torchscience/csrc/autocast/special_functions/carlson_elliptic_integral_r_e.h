#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_TERNARY_AUTOCAST(carlson_elliptic_integral_r_e, x, y, z)
TORCHSCIENCE_TERNARY_AUTOCAST_IMPL(carlson_elliptic_integral_r_e)

} // namespace torchscience::autocast::special_functions
