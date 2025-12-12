#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(cosine_integral_ci)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(cosine_integral_ci)

} // namespace torchscience::autocast::special_functions
