#pragma once

#include "macros.h"

SPARSE_COO_CUDA_UNARY_OPERATOR(special_functions, gamma, z)

SPARSE_COO_CUDA_BINARY_OPERATOR(special_functions, chebyshev_polynomial_t, v, z)

SPARSE_COO_CUDA_TERNARY_OPERATOR(special_functions, incomplete_beta, z, a, b)

SPARSE_COO_CUDA_QUATERNARY_OPERATOR(special_functions, hypergeometric_2_f_1, a, b, c, z)
