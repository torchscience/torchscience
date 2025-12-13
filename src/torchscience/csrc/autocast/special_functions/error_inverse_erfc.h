#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(error_inverse_erfc)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(error_inverse_erfc)

} // namespace torchscience::autocast::special_functions
