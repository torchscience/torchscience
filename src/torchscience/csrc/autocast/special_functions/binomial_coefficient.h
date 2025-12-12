#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(binomial_coefficient, n, k)

TORCHSCIENCE_BINARY_AUTOCAST_IMPL(binomial_coefficient)

} // namespace torchscience::autocast::special_functions
