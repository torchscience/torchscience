#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/euler_number_e.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_UNARY_CPU_KERNEL(euler_number_e)
TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(euler_number_e)

} // namespace torchscience::cpu::special_functions
