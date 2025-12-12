#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(airy_bi)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(airy_bi)

} // namespace torchscience::autocast::special_functions
