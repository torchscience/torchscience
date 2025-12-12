#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(complete_elliptic_integral_e)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(complete_elliptic_integral_e)

} // namespace torchscience::autocast::special_functions
