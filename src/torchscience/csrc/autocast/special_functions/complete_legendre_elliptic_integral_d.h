#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(complete_legendre_elliptic_integral_d)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(complete_legendre_elliptic_integral_d)

} // namespace torchscience::autocast::special_functions
