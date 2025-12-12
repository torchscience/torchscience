#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(bulirsch_elliptic_integral_el1, x, kc)

TORCHSCIENCE_BINARY_AUTOCAST_IMPL(bulirsch_elliptic_integral_el1)

} // namespace torchscience::autocast::special_functions
