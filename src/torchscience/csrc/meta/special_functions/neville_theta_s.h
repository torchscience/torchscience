#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(neville_theta_s, k, u)
TORCHSCIENCE_BINARY_META_KERNEL_IMPL(neville_theta_s)

} // namespace torchscience::meta::special_functions
