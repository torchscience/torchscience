#pragma once

#include <torchscience/csrc/impl/special_functions/chebyshev_polynomial_w.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(chebyshev_polynomial_w, n, x)
TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(chebyshev_polynomial_w)

} // namespace torchscience::quantized::cpu::special_functions
