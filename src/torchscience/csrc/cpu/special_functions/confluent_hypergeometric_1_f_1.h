#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/confluent_hypergeometric_1_f_1.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_TERNARY_CPU_KERNEL(confluent_hypergeometric_1_f_1, a, b, z)
TORCHSCIENCE_TERNARY_CPU_KERNEL_IMPL(confluent_hypergeometric_1_f_1)

} // namespace torchscience::cpu::special_functions
