#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(jacobi_elliptic_dn, u, k)
TORCHSCIENCE_BINARY_META_KERNEL_IMPL(jacobi_elliptic_dn)

} // namespace torchscience::meta::special_functions
