#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL(neville_theta_c, k, u)
TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL_IMPL(neville_theta_c)

} // namespace torchscience::sparse::coo::cpu::special_functions
