#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL(hyperbolic_cosine_integral_chi)

TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL_IMPL(hyperbolic_cosine_integral_chi)

} // namespace torchscience::sparse::csr::cpu::special_functions
