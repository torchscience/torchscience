#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/hankel_h_2.h>

namespace torchscience::cpu::special_functions {

TORCHSCIENCE_BINARY_CPU_KERNEL(hankel_h_2, nu, x)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(hankel_h_2)

} // namespace torchscience::cpu::special_functions
