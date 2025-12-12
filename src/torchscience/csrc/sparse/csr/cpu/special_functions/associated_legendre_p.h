#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_TERNARY_SPARSE_CSR_CPU_KERNEL(associated_legendre_p, n, m, x)
TORCHSCIENCE_TERNARY_SPARSE_CSR_CPU_KERNEL_IMPL(associated_legendre_p)

} // namespace torchscience::sparse::csr::cpu::special_functions
