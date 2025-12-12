#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(spherical_bessel_j, n, x)
TORCHSCIENCE_BINARY_META_KERNEL_IMPL(spherical_bessel_j)

} // namespace torchscience::meta::special_functions
