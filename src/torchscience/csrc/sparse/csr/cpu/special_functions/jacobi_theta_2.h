#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL(jacobi_theta_2, z, q)
TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL_IMPL(jacobi_theta_2)

} // namespace torchscience::sparse::csr::cpu::special_functions
