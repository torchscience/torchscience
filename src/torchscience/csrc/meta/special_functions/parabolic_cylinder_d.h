#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(parabolic_cylinder_d, nu, z)
TORCHSCIENCE_BINARY_META_KERNEL_IMPL(parabolic_cylinder_d)

} // namespace torchscience::meta::special_functions
