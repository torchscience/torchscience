#pragma once

#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_QUATERNARY_QUANTIZED_CPU_KERNEL(carlson_elliptic_integral_r_j, x, y, z, p)
TORCHSCIENCE_QUATERNARY_QUANTIZED_CPU_KERNEL_IMPL(carlson_elliptic_integral_r_j)

} // namespace torchscience::quantized::cpu::special_functions
