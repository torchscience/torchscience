#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL(exponential_integral_e, n, x)
TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL_IMPL(exponential_integral_e)

} // namespace torchscience::sparse::csr::cpu::special_functions
