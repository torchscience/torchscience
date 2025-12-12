#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(hankel_h_2, nu, x)

TORCHSCIENCE_BINARY_META_KERNEL_IMPL(hankel_h_2)

} // namespace torchscience::meta::special_functions
