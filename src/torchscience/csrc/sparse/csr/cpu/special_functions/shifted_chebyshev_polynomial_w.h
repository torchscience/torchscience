#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL(shifted_chebyshev_polynomial_w, n, x)

} // namespace torchscience::sparse::csr::cpu::special_functions
