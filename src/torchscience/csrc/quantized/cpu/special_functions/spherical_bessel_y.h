#pragma once

#include <torchscience/csrc/impl/special_functions/spherical_bessel_y.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(spherical_bessel_y, n, x)
TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(spherical_bessel_y)

} // namespace torchscience::quantized::cpu::special_functions
