#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(binomial_coefficient, n, k)

TORCHSCIENCE_BINARY_META_KERNEL_IMPL(binomial_coefficient)

} // namespace torchscience::meta::special_functions
