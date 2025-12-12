#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(falling_factorial, x, n)

TORCHSCIENCE_BINARY_META_KERNEL_IMPL(falling_factorial)

} // namespace torchscience::meta::special_functions
