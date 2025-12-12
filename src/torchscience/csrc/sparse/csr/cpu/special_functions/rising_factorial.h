#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL(rising_factorial, x, n)

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL_IMPL(rising_factorial)

} // namespace torchscience::sparse::csr::cpu::special_functions
