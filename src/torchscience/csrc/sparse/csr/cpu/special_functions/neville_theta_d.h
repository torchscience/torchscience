#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL(neville_theta_d, k, u)
TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL_IMPL(neville_theta_d)

} // namespace torchscience::sparse::csr::cpu::special_functions
