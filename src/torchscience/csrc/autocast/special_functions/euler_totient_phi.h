#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(euler_totient_phi)
TORCHSCIENCE_UNARY_AUTOCAST_IMPL(euler_totient_phi)

} // namespace torchscience::autocast::special_functions
