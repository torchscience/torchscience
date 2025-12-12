#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(sine_integral_si)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(sine_integral_si)

} // namespace torchscience::autocast::special_functions
