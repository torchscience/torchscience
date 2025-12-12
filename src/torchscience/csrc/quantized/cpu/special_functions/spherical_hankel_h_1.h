#pragma once

#include <torchscience/csrc/impl/special_functions/spherical_hankel_h_1.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(spherical_hankel_h_1, n, x)

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(spherical_hankel_h_1)

} // namespace torchscience::quantized::cpu::special_functions
