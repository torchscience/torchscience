#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/jacobi_theta_4.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_CPU_KERNEL(jacobi_theta_4, z, q)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(jacobi_theta_4)

} // namespace torchscience::cpu::special_functions
