#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(double_factorial)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(double_factorial)

} // namespace torchscience::autocast::special_functions
