#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_TERNARY_AUTOCAST(confluent_hypergeometric_1_f_1, a, b, z)
TORCHSCIENCE_TERNARY_AUTOCAST_IMPL(confluent_hypergeometric_1_f_1)

} // namespace torchscience::autocast::special_functions
