#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_QUATERNARY_META_KERNEL(carlson_elliptic_integral_r_j, x, y, z, p)
TORCHSCIENCE_QUATERNARY_META_KERNEL_IMPL(carlson_elliptic_integral_r_j)

} // namespace torchscience::meta::special_functions
