#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_UNARY_META_KERNEL(factorial)

TORCHSCIENCE_UNARY_META_KERNEL_IMPL(factorial)

} // namespace torchscience::meta::special_functions
