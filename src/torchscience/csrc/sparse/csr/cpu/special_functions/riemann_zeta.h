#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL(riemann_zeta)
TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL_IMPL(riemann_zeta)

} // namespace torchscience::sparse::csr::cpu::special_functions
