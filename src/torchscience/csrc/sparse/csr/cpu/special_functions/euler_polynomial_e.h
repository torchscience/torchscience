#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/euler_polynomial_e.h>

namespace torchscience::sparse::csr::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL(euler_polynomial_e)

} // namespace torchscience::sparse::csr::cpu::special_functions
