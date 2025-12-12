#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(exponential_integral_ei)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(exponential_integral_ei)

} // namespace torchscience::autocast::special_functions
