#pragma once

#include <torchscience/csrc/impl/special_functions/spherical_modified_bessel_i.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(spherical_modified_bessel_i, n, x)
TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(spherical_modified_bessel_i)

} // namespace torchscience::quantized::cpu::special_functions
