#pragma once

#include "operators.cuh"

// Template-based registration (Sparse COO CUDA operators delegate to dense ops)
TORCH_LIBRARY_IMPL(torchscience, SparseCUDA, m_sparse_coo_cuda_special_functions) {
    REGISTER_SPARSE_COO_CUDA_UNARY(m_sparse_coo_cuda_special_functions, gamma);
    REGISTER_SPARSE_COO_CUDA_BINARY(m_sparse_coo_cuda_special_functions, chebyshev_polynomial_t);
    REGISTER_SPARSE_COO_CUDA_TERNARY(m_sparse_coo_cuda_special_functions, incomplete_beta);
    REGISTER_SPARSE_COO_CUDA_QUATERNARY(m_sparse_coo_cuda_special_functions, hypergeometric_2_f_1);
}
