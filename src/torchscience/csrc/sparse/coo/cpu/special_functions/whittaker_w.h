#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_TERNARY_SPARSE_COO_CPU_KERNEL(whittaker_w, kappa, mu, z)
TORCHSCIENCE_TERNARY_SPARSE_COO_CPU_KERNEL_IMPL(whittaker_w)

} // namespace torchscience::sparse::coo::cpu::special_functions
