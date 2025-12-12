#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(fibonacci_number_f)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(fibonacci_number_f)

} // namespace torchscience::autocast::special_functions
