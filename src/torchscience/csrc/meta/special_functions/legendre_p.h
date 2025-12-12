#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(legendre_p, n, x)
TORCHSCIENCE_BINARY_META_KERNEL_IMPL(legendre_p)

} // namespace torchscience::meta::special_functions
