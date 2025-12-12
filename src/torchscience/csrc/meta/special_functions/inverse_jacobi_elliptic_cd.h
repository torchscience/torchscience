#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(inverse_jacobi_elliptic_cd, x, k)
TORCHSCIENCE_BINARY_META_KERNEL_IMPL(inverse_jacobi_elliptic_cd)

} // namespace torchscience::meta::special_functions
