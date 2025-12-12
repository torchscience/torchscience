#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_TERNARY_META_KERNEL(confluent_hypergeometric_1_f_1, a, b, z)
TORCHSCIENCE_TERNARY_META_KERNEL_IMPL(confluent_hypergeometric_1_f_1)

} // namespace torchscience::meta::special_functions
