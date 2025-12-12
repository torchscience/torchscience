#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(bessel_y, nu, x)
TORCHSCIENCE_BINARY_META_KERNEL_IMPL(bessel_y)

} // namespace torchscience::meta::special_functions
