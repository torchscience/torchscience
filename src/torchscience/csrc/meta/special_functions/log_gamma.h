#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_UNARY_META_KERNEL(log_gamma)

TORCHSCIENCE_UNARY_META_KERNEL_IMPL(log_gamma)

} // namespace torchscience::meta::special_functions
