#pragma once

#include "macros.h"

#include "../../kernel/special_functions/chebyshev_polynomial_v.h"
#include "../../kernel/special_functions/chebyshev_polynomial_v_backward.h"
#include "../../kernel/special_functions/chebyshev_polynomial_v_backward_backward.h"

TORCHSCIENCE_CPU_BINARY_OPERATOR(chebyshev_polynomial_v, x, n)
