#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_TERNARY_SPARSE_COO_CPU_KERNEL(associated_legendre_p, n, m, x)
TORCHSCIENCE_TERNARY_SPARSE_COO_CPU_KERNEL_IMPL(associated_legendre_p)

} // namespace torchscience::sparse::coo::cpu::special_functions
