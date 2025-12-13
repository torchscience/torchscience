#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/euler_polynomial_e.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_CPU_KERNEL(euler_polynomial_e, n, x)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(euler_polynomial_e)

} // namespace torchscience::cpu::special_functions
