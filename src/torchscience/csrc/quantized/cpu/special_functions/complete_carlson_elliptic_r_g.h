#pragma once

#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(complete_carlson_elliptic_r_g, x, y)
TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(complete_carlson_elliptic_r_g)

} // namespace torchscience::quantized::cpu::special_functions
