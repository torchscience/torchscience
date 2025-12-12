#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_UNARY_META_KERNEL(double_factorial)

TORCHSCIENCE_UNARY_META_KERNEL_IMPL(double_factorial)

} // namespace torchscience::meta::special_functions
