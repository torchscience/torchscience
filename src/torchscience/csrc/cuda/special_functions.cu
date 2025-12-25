#define TORCH_ASSERT_NO_OPERATORS

#include "../impl/special_functions/chebyshev_polynomial_t.h"
#include "../impl/special_functions/gamma.h"
#include "../impl/special_functions/gamma_backward.h"
#include "../impl/special_functions/gamma_backward_backward.h"
#include "../impl/special_functions/gamma_traits.h"
#include "../impl/special_functions/hypergeometric_2_f_1.h"
#include "../impl/special_functions/hypergeometric_2_f_1_backward.h"
#include "../impl/special_functions/hypergeometric_2_f_1_backward_backward.h"
#include "../impl/special_functions/incomplete_beta.h"
#include "../impl/special_functions/incomplete_beta_backward.h"
#include "../impl/special_functions/incomplete_beta_backward_backward.h"

#include "operators.cuh"
#include "macros.cuh"

using torchscience::impl::special_functions::GammaImpl;

TORCH_LIBRARY_IMPL(torchscience, CUDA, m_cuda_special_functions) {
    REGISTER_CUDA_UNARY(m_cuda_special_functions, gamma, GammaImpl);
}

CUDA_BINARY_OPERATOR(special_functions, chebyshev_polynomial_t, v, z)

CUDA_TERNARY_OPERATOR(special_functions, incomplete_beta, z, a, b)

CUDA_QUATERNARY_OPERATOR(special_functions, hypergeometric_2_f_1, a, b, c, z)
