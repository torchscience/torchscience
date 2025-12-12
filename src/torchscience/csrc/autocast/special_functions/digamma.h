#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(digamma)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(digamma)

} // namespace torchscience::autocast::special_functions
