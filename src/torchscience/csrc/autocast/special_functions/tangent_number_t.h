#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(tangent_number_t)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(tangent_number_t)

} // namespace torchscience::autocast::special_functions
