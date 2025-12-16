#pragma once

#include "macros.h"

SPARSE_COO_CUDA_UNARY_OPERATOR(special_functions, gamma, z)

SPARSE_COO_CUDA_BINARY_OPERATOR(special_functions, chebyshev_polynomial_t, v, z)
