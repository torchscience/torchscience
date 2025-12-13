#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(error_inverse_erf)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(error_inverse_erf)

} // namespace torchscience::autocast::special_functions
