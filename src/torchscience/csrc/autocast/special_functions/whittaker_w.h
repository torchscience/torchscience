#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_TERNARY_AUTOCAST(whittaker_w, kappa, mu, z)
TORCHSCIENCE_TERNARY_AUTOCAST_IMPL(whittaker_w)

} // namespace torchscience::autocast::special_functions
