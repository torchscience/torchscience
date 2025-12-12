#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_QUATERNARY_AUTOCAST(carlson_elliptic_integral_r_j, x, y, z, p)
TORCHSCIENCE_QUATERNARY_AUTOCAST_IMPL(carlson_elliptic_integral_r_j)

} // namespace torchscience::autocast::special_functions
