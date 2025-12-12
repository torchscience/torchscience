#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(bessel_j, nu, x)
TORCHSCIENCE_BINARY_META_KERNEL_IMPL(bessel_j)

} // namespace torchscience::meta::special_functions
