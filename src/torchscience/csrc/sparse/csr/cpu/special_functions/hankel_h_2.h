#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL(hankel_h_2, nu, x)

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL_IMPL(hankel_h_2)

} // namespace torchscience::sparse::csr::cpu::special_functions
