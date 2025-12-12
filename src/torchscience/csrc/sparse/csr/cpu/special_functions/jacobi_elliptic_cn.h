#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL(jacobi_elliptic_cn, u, k)
TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL_IMPL(jacobi_elliptic_cn)

} // namespace torchscience::sparse::csr::cpu::special_functions
