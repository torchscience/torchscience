#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_UNARY_META_KERNEL(inverse_erf)

TORCHSCIENCE_UNARY_META_KERNEL_IMPL(inverse_erf)

} // namespace torchscience::meta::special_functions
