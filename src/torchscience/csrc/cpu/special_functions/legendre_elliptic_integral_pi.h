#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/legendre_elliptic_integral_pi.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_TERNARY_CPU_KERNEL(legendre_elliptic_integral_pi, n, phi, k)
TORCHSCIENCE_TERNARY_CPU_KERNEL_IMPL(legendre_elliptic_integral_pi)

} // namespace torchscience::cpu::special_functions
