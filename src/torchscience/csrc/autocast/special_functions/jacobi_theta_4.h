#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(jacobi_theta_4, z, q)
TORCHSCIENCE_BINARY_AUTOCAST_IMPL(jacobi_theta_4)

} // namespace torchscience::autocast::special_functions
