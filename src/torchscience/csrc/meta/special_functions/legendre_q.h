#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(legendre_q, n, x)
TORCHSCIENCE_BINARY_META_KERNEL_IMPL(legendre_q)

} // namespace torchscience::meta::special_functions
