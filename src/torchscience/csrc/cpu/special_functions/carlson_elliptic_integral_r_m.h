#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/carlson_elliptic_integral_r_m.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_TERNARY_CPU_KERNEL(carlson_elliptic_integral_r_m, x, y, z)
TORCHSCIENCE_TERNARY_CPU_KERNEL_IMPL(carlson_elliptic_integral_r_m)

} // namespace torchscience::cpu::special_functions
