#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_UNARY_META_KERNEL(euler_totient_phi)
TORCHSCIENCE_UNARY_META_KERNEL_IMPL(euler_totient_phi)

} // namespace torchscience::meta::special_functions
