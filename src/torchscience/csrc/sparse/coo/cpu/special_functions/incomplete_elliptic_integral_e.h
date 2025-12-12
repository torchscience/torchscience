#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL(incomplete_elliptic_integral_e, phi, k)
TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL_IMPL(incomplete_elliptic_integral_e)

} // namespace torchscience::sparse::coo::cpu::special_functions
