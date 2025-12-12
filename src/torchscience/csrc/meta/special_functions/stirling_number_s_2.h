#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(stirling_number_s_2, n, k)
TORCHSCIENCE_BINARY_META_KERNEL_IMPL(stirling_number_s_2)

} // namespace torchscience::meta::special_functions
