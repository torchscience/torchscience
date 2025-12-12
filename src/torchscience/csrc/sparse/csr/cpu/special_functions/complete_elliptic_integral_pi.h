#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL(complete_elliptic_integral_pi, n, k)
TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL_IMPL(complete_elliptic_integral_pi)

} // namespace torchscience::sparse::csr::cpu::special_functions
