#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL(complete_legendre_elliptic_integral_d)

TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL_IMPL(complete_legendre_elliptic_integral_d)

} // namespace torchscience::sparse::coo::cpu::special_functions
