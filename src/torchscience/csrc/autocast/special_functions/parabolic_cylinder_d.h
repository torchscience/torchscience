#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(parabolic_cylinder_d, nu, z)
TORCHSCIENCE_BINARY_AUTOCAST_IMPL(parabolic_cylinder_d)

} // namespace torchscience::autocast::special_functions
