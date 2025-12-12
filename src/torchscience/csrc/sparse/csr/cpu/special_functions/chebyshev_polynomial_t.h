#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL(chebyshev_polynomial_t, n, x)
TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL_IMPL(chebyshev_polynomial_t)

} // namespace torchscience::sparse::csr::cpu::special_functions
