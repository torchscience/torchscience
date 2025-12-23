#pragma once

#include "operators.h"

// Template-based registration (Sparse COO CPU operators delegate to dense ops)
TORCH_LIBRARY_IMPL(torchscience, SparseCPU, m_sparse_coo_cpu_special_functions) {
    REGISTER_SPARSE_COO_CPU_UNARY(m_sparse_coo_cpu_special_functions, gamma);
    REGISTER_SPARSE_COO_CPU_BINARY(m_sparse_coo_cpu_special_functions, chebyshev_polynomial_t);
    REGISTER_SPARSE_COO_CPU_TERNARY(m_sparse_coo_cpu_special_functions, incomplete_beta);
    REGISTER_SPARSE_COO_CPU_QUATERNARY(m_sparse_coo_cpu_special_functions, hypergeometric_2_f_1);
}
