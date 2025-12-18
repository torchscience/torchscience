#define TORCH_ASSERT_NO_OPERATORS

#include "../impl/special_functions/chebyshev_polynomial_t.h"
#include "../impl/special_functions/gamma.h"
#include "../impl/special_functions/incomplete_beta.h"

#include "macros.cuh"

CUDA_UNARY_OPERATOR(special_functions, gamma, z)

CUDA_BINARY_OPERATOR(special_functions, chebyshev_polynomial_t, v, z)

CUDA_TERNARY_OPERATOR(special_functions, incomplete_beta, z, a, b)
