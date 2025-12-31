#pragma once

#include "macros.h"

#include "../kernel/special_functions/gamma.h"
#include "../kernel/special_functions/gamma_backward.h"
#include "../kernel/special_functions/gamma_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR(gamma, z)

#include "../kernel/special_functions/digamma.h"
#include "../kernel/special_functions/digamma_backward.h"
#include "../kernel/special_functions/digamma_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR(digamma, z)

#include "../kernel/special_functions/beta.h"
#include "../kernel/special_functions/beta_backward.h"
#include "../kernel/special_functions/beta_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR(beta, a, b)

#include "../kernel/special_functions/chebyshev_polynomial_t.h"
#include "../kernel/special_functions/chebyshev_polynomial_t_backward.h"
#include "../kernel/special_functions/chebyshev_polynomial_t_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR(chebyshev_polynomial_t, x, n)

#include "../kernel/special_functions/incomplete_beta.h"
#include "../kernel/special_functions/incomplete_beta_backward.h"
#include "../kernel/special_functions/incomplete_beta_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_TERNARY_OPERATOR(incomplete_beta, x, a, b)

#include "../kernel/special_functions/hypergeometric_2_f_1.h"
#include "../kernel/special_functions/hypergeometric_2_f_1_backward.h"
#include "../kernel/special_functions/hypergeometric_2_f_1_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_QUATERNARY_OPERATOR(hypergeometric_2_f_1, a, b, c, z)
