#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(bessel_j_derivative, nu, x)

TORCHSCIENCE_BINARY_META_KERNEL_IMPL(bessel_j_derivative)

} // namespace torchscience::meta::special_functions
