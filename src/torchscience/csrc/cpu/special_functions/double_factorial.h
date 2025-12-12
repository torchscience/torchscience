#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/double_factorial.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_UNARY_CPU_KERNEL(double_factorial)
TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(double_factorial)

} // namespace torchscience::cpu::special_functions
