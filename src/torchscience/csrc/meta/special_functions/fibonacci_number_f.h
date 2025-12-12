#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_UNARY_META_KERNEL(fibonacci_number_f)

TORCHSCIENCE_UNARY_META_KERNEL_IMPL(fibonacci_number_f)

} // namespace torchscience::meta::special_functions
