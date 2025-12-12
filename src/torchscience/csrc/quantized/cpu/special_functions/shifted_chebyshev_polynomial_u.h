#pragma once

#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(shifted_chebyshev_polynomial_u, n, x)

} // namespace torchscience::quantized::cpu::special_functions
