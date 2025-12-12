#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL(hyperbolic_sine_integral_shi)

TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL_IMPL(hyperbolic_sine_integral_shi)

} // namespace torchscience::sparse::csr::cpu::special_functions
