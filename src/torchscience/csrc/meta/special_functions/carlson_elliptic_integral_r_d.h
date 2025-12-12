#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_TERNARY_META_KERNEL(carlson_elliptic_integral_r_d, x, y, z)
TORCHSCIENCE_TERNARY_META_KERNEL_IMPL(carlson_elliptic_integral_r_d)

} // namespace torchscience::meta::special_functions
