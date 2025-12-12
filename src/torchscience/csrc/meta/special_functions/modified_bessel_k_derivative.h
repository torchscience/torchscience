#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(modified_bessel_k_derivative, nu, x)

TORCHSCIENCE_BINARY_META_KERNEL_IMPL(modified_bessel_k_derivative)

} // namespace torchscience::meta::special_functions
