#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(spherical_hankel_h_1, n, x)

TORCHSCIENCE_BINARY_META_KERNEL_IMPL(spherical_hankel_h_1)

} // namespace torchscience::meta::special_functions
