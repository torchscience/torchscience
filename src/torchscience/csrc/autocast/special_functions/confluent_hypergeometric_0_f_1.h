#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(confluent_hypergeometric_0_f_1, b, z)

TORCHSCIENCE_BINARY_AUTOCAST_IMPL(confluent_hypergeometric_0_f_1)

} // namespace torchscience::autocast::special_functions
