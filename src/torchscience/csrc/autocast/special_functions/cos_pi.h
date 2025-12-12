#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(cos_pi)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(cos_pi)

} // namespace torchscience::autocast::special_functions
