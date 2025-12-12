#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/jacobi_theta_3.h>

namespace torchscience::cpu::special_functions {

TORCHSCIENCE_BINARY_CPU_KERNEL(jacobi_theta_3, z, q)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(jacobi_theta_3)

} // namespace torchscience::cpu::special_functions
