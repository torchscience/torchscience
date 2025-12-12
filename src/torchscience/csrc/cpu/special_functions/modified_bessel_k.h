#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/modified_bessel_k.h>

namespace torchscience::cpu::special_functions {

TORCHSCIENCE_BINARY_CPU_KERNEL(modified_bessel_k, nu, x)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(modified_bessel_k)

} // namespace torchscience::cpu::special_functions
