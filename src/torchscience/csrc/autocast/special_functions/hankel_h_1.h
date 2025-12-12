#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(hankel_h_1, nu, x)

TORCHSCIENCE_BINARY_AUTOCAST_IMPL(hankel_h_1)

} // namespace torchscience::autocast::special_functions
