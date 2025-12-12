#pragma once

#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(jacobi_theta_2, z, q)
TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(jacobi_theta_2)

} // namespace torchscience::quantized::cpu::special_functions
