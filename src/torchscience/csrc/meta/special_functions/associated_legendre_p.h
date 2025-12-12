#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_TERNARY_META_KERNEL(associated_legendre_p, n, m, x)
TORCHSCIENCE_TERNARY_META_KERNEL_IMPL(associated_legendre_p)

} // namespace torchscience::meta::special_functions
