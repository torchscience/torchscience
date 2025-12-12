#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(erf)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(erf)

} // namespace torchscience::autocast::special_functions
