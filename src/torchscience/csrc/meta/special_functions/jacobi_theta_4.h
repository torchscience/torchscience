#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(jacobi_theta_4, z, q)
TORCHSCIENCE_BINARY_META_KERNEL_IMPL(jacobi_theta_4)

} // namespace torchscience::meta::special_functions
