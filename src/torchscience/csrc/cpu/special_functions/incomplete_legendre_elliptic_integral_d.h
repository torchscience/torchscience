#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/incomplete_legendre_elliptic_integral_d.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_CPU_KERNEL(incomplete_legendre_elliptic_integral_d, phi, k)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(incomplete_legendre_elliptic_integral_d)

} // namespace torchscience::cpu::special_functions
