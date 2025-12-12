#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_TERNARY_SPARSE_CSR_CPU_KERNEL(confluent_hypergeometric_1_f_1, a, b, z)
TORCHSCIENCE_TERNARY_SPARSE_CSR_CPU_KERNEL_IMPL(confluent_hypergeometric_1_f_1)

} // namespace torchscience::sparse::csr::cpu::special_functions
