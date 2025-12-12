#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL(bessel_j_derivative, nu, x)

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL_IMPL(bessel_j_derivative)

} // namespace torchscience::sparse::csr::cpu::special_functions
