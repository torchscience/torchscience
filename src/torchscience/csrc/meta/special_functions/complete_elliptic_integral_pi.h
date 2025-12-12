#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(complete_elliptic_integral_pi, n, k)
TORCHSCIENCE_BINARY_META_KERNEL_IMPL(complete_elliptic_integral_pi)

} // namespace torchscience::meta::special_functions
