#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/fibonacci_number_f.h>

namespace torchscience::cpu::special_functions {

TORCHSCIENCE_UNARY_CPU_KERNEL(fibonacci_number_f)
TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(fibonacci_number_f)

} // namespace torchscience::cpu::special_functions
