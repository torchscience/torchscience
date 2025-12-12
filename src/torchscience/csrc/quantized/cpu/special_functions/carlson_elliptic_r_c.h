#pragma once

#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(carlson_elliptic_r_c, x, y)
TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(carlson_elliptic_r_c)

} // namespace torchscience::quantized::cpu::special_functions
