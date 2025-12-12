#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(sine_integral_sin)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(sine_integral_sin)

} // namespace torchscience::autocast::special_functions
