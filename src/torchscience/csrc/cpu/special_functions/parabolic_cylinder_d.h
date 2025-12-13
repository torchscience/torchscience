#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/parabolic_cylinder_d.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_CPU_KERNEL(parabolic_cylinder_d, nu, z)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(parabolic_cylinder_d)

} // namespace torchscience::cpu::special_functions
