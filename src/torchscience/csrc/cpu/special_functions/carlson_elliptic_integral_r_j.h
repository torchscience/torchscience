#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/carlson_elliptic_integral_r_j.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_QUATERNARY_CPU_KERNEL(carlson_elliptic_integral_r_j, x, y, z, p)
TORCHSCIENCE_QUATERNARY_CPU_KERNEL_IMPL(carlson_elliptic_integral_r_j)

} // namespace torchscience::cpu::special_functions
