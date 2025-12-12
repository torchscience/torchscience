#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL(incomplete_legendre_elliptic_integral_d, phi, k)
TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL_IMPL(incomplete_legendre_elliptic_integral_d)

} // namespace torchscience::sparse::csr::cpu::special_functions
