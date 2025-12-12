#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_UNARY_META_KERNEL(inverse_erfc)

TORCHSCIENCE_UNARY_META_KERNEL_IMPL(inverse_erfc)

} // namespace torchscience::meta::special_functions
