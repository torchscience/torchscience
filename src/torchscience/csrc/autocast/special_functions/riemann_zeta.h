#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(riemann_zeta)
TORCHSCIENCE_UNARY_AUTOCAST_IMPL(riemann_zeta)

} // namespace torchscience::autocast::special_functions
