#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_UNARY_AUTOCAST(prime_number_p)

TORCHSCIENCE_UNARY_AUTOCAST_IMPL(prime_number_p)

} // namespace torchscience::autocast::special_functions
