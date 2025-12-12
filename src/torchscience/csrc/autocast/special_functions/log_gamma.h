#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(log_gamma)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(log_gamma)

} // namespace torchscience::autocast::special_functions
