#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL(binomial_coefficient, n, k)

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL_IMPL(binomial_coefficient)

} // namespace torchscience::sparse::csr::cpu::special_functions
