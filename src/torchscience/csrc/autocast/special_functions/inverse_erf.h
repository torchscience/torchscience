#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(inverse_erf)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(inverse_erf)

} // namespace torchscience::autocast::special_functions
