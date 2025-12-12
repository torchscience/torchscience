#pragma once

#include <torchscience/csrc/impl/special_functions/inverse_jacobi_elliptic_sc.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(inverse_jacobi_elliptic_sc, x, k)
TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(inverse_jacobi_elliptic_sc)

} // namespace torchscience::quantized::cpu::special_functions
