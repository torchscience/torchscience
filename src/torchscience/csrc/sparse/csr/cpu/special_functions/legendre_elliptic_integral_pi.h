#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_TERNARY_SPARSE_CSR_CPU_KERNEL(legendre_elliptic_integral_pi, n, phi, k)
TORCHSCIENCE_TERNARY_SPARSE_CSR_CPU_KERNEL_IMPL(legendre_elliptic_integral_pi)

} // namespace torchscience::sparse::csr::cpu::special_functions
