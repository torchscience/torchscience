#pragma once

#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(parabolic_cylinder_d, nu, z)
TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(parabolic_cylinder_d)

} // namespace torchscience::quantized::cpu::special_functions
