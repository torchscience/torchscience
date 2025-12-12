#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(jacobi_elliptic_sn, u, k)
TORCHSCIENCE_BINARY_META_KERNEL_IMPL(jacobi_elliptic_sn)

} // namespace torchscience::meta::special_functions
