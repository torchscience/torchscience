#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL(incomplete_elliptic_integral_f, phi, k)
TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL_IMPL(incomplete_elliptic_integral_f)

} // namespace torchscience::sparse::csr::cpu::special_functions
