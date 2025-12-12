#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL(spherical_hankel_h_1, n, x)

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL_IMPL(spherical_hankel_h_1)

} // namespace torchscience::sparse::csr::cpu::special_functions
