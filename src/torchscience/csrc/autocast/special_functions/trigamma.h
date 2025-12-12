#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(trigamma)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(trigamma)

} // namespace torchscience::autocast::special_functions
