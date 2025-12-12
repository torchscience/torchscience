#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL(stirling_number_s_2, n, k)
TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL_IMPL(stirling_number_s_2)

} // namespace torchscience::sparse::coo::cpu::special_functions
