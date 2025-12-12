#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(log_beta, a, b)

TORCHSCIENCE_BINARY_AUTOCAST_IMPL(log_beta)

} // namespace torchscience::autocast::special_functions
