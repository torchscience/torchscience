#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/falling_factorial.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_CPU_KERNEL(falling_factorial, x, n)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(falling_factorial)

} // namespace torchscience::cpu::special_functions
