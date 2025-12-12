#pragma once

#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_TERNARY_QUANTIZED_CPU_KERNEL(carlson_elliptic_integral_r_d, x, y, z)
TORCHSCIENCE_TERNARY_QUANTIZED_CPU_KERNEL_IMPL(carlson_elliptic_integral_r_d)

} // namespace torchscience::quantized::cpu::special_functions
