#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(liouville_lambda)
TORCHSCIENCE_UNARY_AUTOCAST_IMPL(liouville_lambda)

} // namespace torchscience::autocast::special_functions
