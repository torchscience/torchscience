#pragma once

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

#include "macros.h"

CPU_UNARY_OPERATOR(special_functions, gamma, z)

CPU_BINARY_OPERATOR(special_functions, chebyshev_polynomial_t, v, z)

CPU_TERNARY_OPERATOR(special_functions, incomplete_beta, z, a, b)

CPU_QUATERNARY_OPERATOR(special_functions, hypergeometric_2_f_1, a, b, c, z)
