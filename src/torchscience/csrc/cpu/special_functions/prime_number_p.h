#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/prime_number_p.h>

namespace torchscience::cpu::special_functions {

TORCHSCIENCE_UNARY_CPU_KERNEL(prime_number_p)
TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(prime_number_p)

} // namespace torchscience::cpu::special_functions
