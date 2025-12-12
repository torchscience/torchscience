#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL(confluent_hypergeometric_0_f_1, b, z)

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL_IMPL(confluent_hypergeometric_0_f_1)

} // namespace torchscience::sparse::csr::cpu::special_functions
