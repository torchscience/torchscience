#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/factorial.h>

namespace torchscience::cpu::special_functions {

TORCHSCIENCE_UNARY_CPU_KERNEL(factorial)
TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(factorial)

} // namespace torchscience::cpu::special_functions
