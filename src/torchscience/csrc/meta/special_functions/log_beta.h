#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(log_beta, a, b)

TORCHSCIENCE_BINARY_META_KERNEL_IMPL(log_beta)

} // namespace torchscience::meta::special_functions
