#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(jacobi_elliptic_sd, u, k)
TORCHSCIENCE_BINARY_AUTOCAST_IMPL(jacobi_elliptic_sd)

} // namespace torchscience::autocast::special_functions
