#pragma once

#include <torchscience/csrc/impl/special_functions/bessel_y.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(bessel_y, nu, x)
TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(bessel_y)

} // namespace torchscience::quantized::cpu::special_functions
