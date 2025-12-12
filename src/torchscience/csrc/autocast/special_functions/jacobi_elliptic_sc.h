#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(jacobi_elliptic_sc, u, k)
TORCHSCIENCE_BINARY_AUTOCAST_IMPL(jacobi_elliptic_sc)

} // namespace torchscience::autocast::special_functions
