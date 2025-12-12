#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(chebyshev_polynomial_u, n, x)
TORCHSCIENCE_BINARY_META_KERNEL_IMPL(chebyshev_polynomial_u)

} // namespace torchscience::meta::special_functions
