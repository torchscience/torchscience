#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_TERNARY_META_KERNEL(legendre_elliptic_integral_pi, n, phi, k)
TORCHSCIENCE_TERNARY_META_KERNEL_IMPL(legendre_elliptic_integral_pi)

} // namespace torchscience::meta::special_functions
