#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/exponential_integral_e.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_CPU_KERNEL(exponential_integral_e, n, x)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(exponential_integral_e)

} // namespace torchscience::cpu::special_functions
