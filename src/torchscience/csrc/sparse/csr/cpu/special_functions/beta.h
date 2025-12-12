#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL(beta, a, b)
TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL_IMPL(beta)

} // namespace torchscience::sparse::csr::cpu::special_functions
