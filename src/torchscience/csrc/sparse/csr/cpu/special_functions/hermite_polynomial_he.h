#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL(hermite_polynomial_he, n, x)

} // namespace torchscience::sparse::csr::cpu::special_functions
