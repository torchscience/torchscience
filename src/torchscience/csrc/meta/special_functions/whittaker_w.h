#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_TERNARY_META_KERNEL(whittaker_w, kappa, mu, z)
TORCHSCIENCE_TERNARY_META_KERNEL_IMPL(whittaker_w)

} // namespace torchscience::meta::special_functions
