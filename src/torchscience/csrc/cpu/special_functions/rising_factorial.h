#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/rising_factorial.h>

namespace torchscience::cpu::special_functions {

TORCHSCIENCE_BINARY_CPU_KERNEL(rising_factorial, x, n)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(rising_factorial)

} // namespace torchscience::cpu::special_functions
