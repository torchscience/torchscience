#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(airy_ai)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(airy_ai)

} // namespace torchscience::autocast::special_functions
