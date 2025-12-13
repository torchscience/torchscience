#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/bernoulli_polynomial_b.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_CPU_KERNEL(bernoulli_polynomial_b, n, x)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(bernoulli_polynomial_b)

} // namespace torchscience::cpu::special_functions
