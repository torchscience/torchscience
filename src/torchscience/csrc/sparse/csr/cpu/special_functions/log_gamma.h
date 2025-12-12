#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL(log_gamma)

TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL_IMPL(log_gamma)

} // namespace torchscience::sparse::csr::cpu::special_functions
