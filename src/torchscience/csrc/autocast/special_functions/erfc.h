#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(erfc)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(erfc)

} // namespace torchscience::autocast::special_functions
