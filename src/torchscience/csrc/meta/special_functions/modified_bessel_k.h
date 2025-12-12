#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(modified_bessel_k, nu, x)
TORCHSCIENCE_BINARY_META_KERNEL_IMPL(modified_bessel_k)

} // namespace torchscience::meta::special_functions
