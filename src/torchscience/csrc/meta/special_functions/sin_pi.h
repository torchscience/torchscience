#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_UNARY_META_KERNEL(sin_pi)

TORCHSCIENCE_UNARY_META_KERNEL_IMPL(sin_pi)

} // namespace torchscience::meta::special_functions
