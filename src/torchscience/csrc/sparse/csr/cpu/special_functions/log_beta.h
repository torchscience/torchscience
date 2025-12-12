#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL(log_beta, a, b)

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL_IMPL(log_beta)

} // namespace torchscience::sparse::csr::cpu::special_functions
