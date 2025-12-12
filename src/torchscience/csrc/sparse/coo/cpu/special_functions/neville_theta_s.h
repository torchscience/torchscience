#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL(neville_theta_s, k, u)
TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL_IMPL(neville_theta_s)

} // namespace torchscience::sparse::coo::cpu::special_functions
