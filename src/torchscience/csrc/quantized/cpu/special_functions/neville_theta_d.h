#pragma once

#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(neville_theta_d, k, u)
TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(neville_theta_d)

} // namespace torchscience::quantized::cpu::special_functions
