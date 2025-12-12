#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(modified_bessel_i, nu, x)
TORCHSCIENCE_BINARY_META_KERNEL_IMPL(modified_bessel_i)

} // namespace torchscience::meta::special_functions
