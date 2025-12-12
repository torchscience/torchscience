#pragma once

#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_TERNARY_QUANTIZED_CPU_KERNEL(legendre_elliptic_integral_pi, n, phi, k)
TORCHSCIENCE_TERNARY_QUANTIZED_CPU_KERNEL_IMPL(legendre_elliptic_integral_pi)

} // namespace torchscience::quantized::cpu::special_functions
