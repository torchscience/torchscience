#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL(legendre_p, n, x)
TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL_IMPL(legendre_p)

} // namespace torchscience::sparse::coo::cpu::special_functions
