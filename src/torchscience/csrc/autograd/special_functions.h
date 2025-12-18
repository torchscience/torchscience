#pragma once

#include "macros.h"

AUTOGRAD_UNARY_OPERATOR(special_functions, Gamma, gamma, z)

AUTOGRAD_BINARY_OPERATOR(special_functions, ChebyshevPolynomialT, chebyshev_polynomial_t, v, z)

AUTOGRAD_TERNARY_OPERATOR(special_functions, IncompleteBeta, incomplete_beta, z, a, b)

AUTOGRAD_QUATERNARY_OPERATOR(special_functions, Hypergeometric2F1, hypergeometric_2_f_1, a, b, c, z)
