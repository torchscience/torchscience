#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/bernoulli_polynomial_b.h>

namespace torchscience::sparse::csr::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL(bernoulli_polynomial_b)

} // namespace torchscience::sparse::csr::cpu::special_functions
