#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/kelvin_kei.h>

namespace torchscience::sparse::csr::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL(kelvin_kei)

} // namespace torchscience::sparse::csr::cpu::special_functions
