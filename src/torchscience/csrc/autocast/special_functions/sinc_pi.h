#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(sinc_pi)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(sinc_pi)

} // namespace torchscience::autocast::special_functions
