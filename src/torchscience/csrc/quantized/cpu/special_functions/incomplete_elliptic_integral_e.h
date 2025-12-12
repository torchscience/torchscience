#pragma once

#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(incomplete_elliptic_integral_e, phi, k)
TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(incomplete_elliptic_integral_e)

} // namespace torchscience::quantized::cpu::special_functions
