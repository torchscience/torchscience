#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/binomial_coefficient.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_CPU_KERNEL(binomial_coefficient, n, k)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(binomial_coefficient)

} // namespace torchscience::cpu::special_functions
