#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(airy_bi_derivative)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(airy_bi_derivative)

} // namespace torchscience::autocast::special_functions
