#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(exponential_integral_e_1)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(exponential_integral_e_1)

} // namespace torchscience::autocast::special_functions
