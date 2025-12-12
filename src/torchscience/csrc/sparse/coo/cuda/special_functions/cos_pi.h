#pragma once

#include <torchscience/csrc/sparse/coo/cuda/macros.h>

namespace torchscience::sparse::coo::cuda::special_functions {

TORCHSCIENCE_UNARY_SPARSE_COO_CUDA_KERNEL(cos_pi)

TORCHSCIENCE_UNARY_SPARSE_COO_CUDA_KERNEL_IMPL(cos_pi)

} // namespace torchscience::sparse::coo::cuda::special_functions
