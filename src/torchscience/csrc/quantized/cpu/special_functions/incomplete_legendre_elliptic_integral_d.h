#pragma once

#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(incomplete_legendre_elliptic_integral_d, phi, k)
TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(incomplete_legendre_elliptic_integral_d)

} // namespace torchscience::quantized::cpu::special_functions
