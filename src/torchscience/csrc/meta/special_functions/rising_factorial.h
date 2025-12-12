#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(rising_factorial, x, n)

TORCHSCIENCE_BINARY_META_KERNEL_IMPL(rising_factorial)

} // namespace torchscience::meta::special_functions
