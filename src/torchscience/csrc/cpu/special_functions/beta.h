#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/beta.h>

namespace torchscience::cpu::special_functions {

TORCHSCIENCE_BINARY_CPU_KERNEL(beta, a, b)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(beta)

} // namespace torchscience::cpu::special_functions
