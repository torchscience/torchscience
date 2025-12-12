#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(confluent_hypergeometric_0_f_1, b, z)

TORCHSCIENCE_BINARY_META_KERNEL_IMPL(confluent_hypergeometric_0_f_1)

} // namespace torchscience::meta::special_functions
