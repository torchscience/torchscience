#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/hankel_h_1.h>

namespace torchscience::cpu::special_functions {

TORCHSCIENCE_BINARY_CPU_KERNEL(hankel_h_1, nu, x)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(hankel_h_1)

} // namespace torchscience::cpu::special_functions
