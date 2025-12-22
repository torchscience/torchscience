#define TORCH_ASSERT_NO_OPERATORS

#include "../impl/special_functions/chebyshev_polynomial_t.h"
#include "../impl/special_functions/gamma.h"
#include "../impl/special_functions/gamma_backward.h"
#include "../impl/special_functions/gamma_backward_backward.h"
#include "../impl/special_functions/hypergeometric_2_f_1.h"
#include "../impl/special_functions/hypergeometric_2_f_1_backward.h"
#include "../impl/special_functions/hypergeometric_2_f_1_backward_backward.h"
#include "../impl/special_functions/incomplete_beta.h"
#include "../impl/special_functions/incomplete_beta_backward.h"
#include "../impl/special_functions/incomplete_beta_backward_backward.h"

#include "macros.cuh"

CUDA_UNARY_OPERATOR(special_functions, gamma, z)

CUDA_BINARY_OPERATOR(special_functions, chebyshev_polynomial_t, v, z)

CUDA_TERNARY_OPERATOR(special_functions, incomplete_beta, z, a, b)

CUDA_QUATERNARY_OPERATOR(special_functions, hypergeometric_2_f_1, a, b, c, z)
