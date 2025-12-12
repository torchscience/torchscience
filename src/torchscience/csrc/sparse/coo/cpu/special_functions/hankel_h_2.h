#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL(hankel_h_2, nu, x)

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL_IMPL(hankel_h_2)

} // namespace torchscience::sparse::coo::cpu::special_functions
