#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(mobius_mu)
TORCHSCIENCE_UNARY_AUTOCAST_IMPL(mobius_mu)

} // namespace torchscience::autocast::special_functions
