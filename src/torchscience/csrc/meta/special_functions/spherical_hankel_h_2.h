#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(spherical_hankel_h_2, n, x)

TORCHSCIENCE_BINARY_META_KERNEL_IMPL(spherical_hankel_h_2)

} // namespace torchscience::meta::special_functions
