#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/tangent_number_t_2.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_UNARY_CPU_KERNEL(tangent_number_t_2)
TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(tangent_number_t_2)

} // namespace torchscience::cpu::special_functions
