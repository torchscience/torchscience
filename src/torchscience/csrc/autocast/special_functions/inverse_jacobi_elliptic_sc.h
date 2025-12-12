#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(inverse_jacobi_elliptic_sc, x, k)
TORCHSCIENCE_BINARY_AUTOCAST_IMPL(inverse_jacobi_elliptic_sc)

} // namespace torchscience::autocast::special_functions
