#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(beta, a, b)
TORCHSCIENCE_BINARY_META_KERNEL_IMPL(beta)

} // namespace torchscience::meta::special_functions
