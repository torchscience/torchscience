#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/chebyshev_polynomial_t.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_CPU_KERNEL(chebyshev_polynomial_t, n, x)
TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(chebyshev_polynomial_t)

} // namespace torchscience::cpu::special_functions
