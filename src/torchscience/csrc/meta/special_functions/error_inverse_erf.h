#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_UNARY_META_KERNEL(error_inverse_erf)

TORCHSCIENCE_UNARY_META_KERNEL_IMPL(error_inverse_erf)

} // namespace torchscience::meta::special_functions
