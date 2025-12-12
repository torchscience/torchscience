#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(logarithmic_integral_li)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(logarithmic_integral_li)

} // namespace torchscience::autocast::special_functions
