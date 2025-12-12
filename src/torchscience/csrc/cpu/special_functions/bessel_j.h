#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/bessel_j.h>

namespace torchscience::cpu::special_functions {

TORCHSCIENCE_BINARY_CPU_KERNEL(bessel_j, nu, x)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(bessel_j)

} // namespace torchscience::cpu::special_functions
