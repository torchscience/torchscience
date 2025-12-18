#pragma once

#include "macros.h"

QUANTIZED_CUDA_UNARY_OPERATOR(special_functions, gamma, z)

QUANTIZED_CUDA_BINARY_OPERATOR(special_functions, chebyshev_polynomial_t, v, z)

QUANTIZED_CUDA_TERNARY_OPERATOR(special_functions, incomplete_beta, z, a, b)
