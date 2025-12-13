#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(euler_number_e)
TORCHSCIENCE_UNARY_AUTOCAST_IMPL(euler_number_e)

} // namespace torchscience::autocast::special_functions
