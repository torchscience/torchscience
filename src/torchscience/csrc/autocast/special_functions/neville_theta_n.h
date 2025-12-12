#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(neville_theta_n, k, u)
TORCHSCIENCE_BINARY_AUTOCAST_IMPL(neville_theta_n)

} // namespace torchscience::autocast::special_functions
