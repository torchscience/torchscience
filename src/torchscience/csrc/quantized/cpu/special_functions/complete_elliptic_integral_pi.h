#pragma once

#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(complete_elliptic_integral_pi, n, k)
TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(complete_elliptic_integral_pi)

} // namespace torchscience::quantized::cpu::special_functions
