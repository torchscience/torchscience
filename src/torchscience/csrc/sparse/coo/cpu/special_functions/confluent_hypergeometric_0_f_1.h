#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL(confluent_hypergeometric_0_f_1, b, z)

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL_IMPL(confluent_hypergeometric_0_f_1)

} // namespace torchscience::sparse::coo::cpu::special_functions
