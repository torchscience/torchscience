#pragma once

#include "macros.h"

#include "../../kernel/special_functions/incomplete_beta.h"
#include "../../kernel/special_functions/incomplete_beta_backward.h"
#include "../../kernel/special_functions/incomplete_beta_backward_backward.h"

TORCHSCIENCE_CPU_TERNARY_OPERATOR(incomplete_beta, x, a, b)
