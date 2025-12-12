#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(gamma)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(gamma)

} // namespace torchscience::autocast::special_functions
