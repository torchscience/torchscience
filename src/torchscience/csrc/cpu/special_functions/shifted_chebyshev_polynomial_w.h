#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/shifted_chebyshev_polynomial_w.h>

namespace torchscience::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_CPU_KERNEL(shifted_chebyshev_polynomial_w, n, x)

TORCHSCIENCE_BINARY_CPU_KERNEL_IMPL(shifted_chebyshev_polynomial_w)

} // namespace torchscience::cpu::special_functions
