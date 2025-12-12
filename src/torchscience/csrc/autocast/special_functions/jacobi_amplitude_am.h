#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(jacobi_amplitude_am, u, k)
TORCHSCIENCE_BINARY_AUTOCAST_IMPL(jacobi_amplitude_am)

} // namespace torchscience::autocast::special_functions
