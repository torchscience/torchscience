#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(bernoulli_number_b)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(bernoulli_number_b)

} // namespace torchscience::autocast::special_functions
