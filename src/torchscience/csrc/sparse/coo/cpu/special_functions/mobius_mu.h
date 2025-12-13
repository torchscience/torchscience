#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/mobius_mu.h>

namespace torchscience::sparse::coo::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL(mobius_mu)
TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL_IMPL(mobius_mu)

} // namespace torchscience::sparse::coo::cpu::special_functions
