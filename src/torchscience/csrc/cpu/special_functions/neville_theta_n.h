#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/neville_theta_n.h>

namespace torchscience::cpu::special_functions {

TORCHSCIENCE_BINARY_CPU_KERNEL(neville_theta_n, k, u)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(neville_theta_n)

} // namespace torchscience::cpu::special_functions
