#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(spherical_modified_bessel_i, n, x)
TORCHSCIENCE_BINARY_META_KERNEL_IMPL(spherical_modified_bessel_i)

} // namespace torchscience::meta::special_functions
