#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(exponential_integral_e, n, x)
TORCHSCIENCE_BINARY_META_KERNEL_IMPL(exponential_integral_e)

} // namespace torchscience::meta::special_functions
