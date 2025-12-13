#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(error_erfc)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(error_erfc)

} // namespace torchscience::autocast::special_functions
