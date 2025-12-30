#pragma once

#include "macros.h"

#include "../../kernel/special_functions/gamma.h"
#include "../../kernel/special_functions/gamma_backward.h"
#include "../../kernel/special_functions/gamma_backward_backward.h"

TORCHSCIENCE_CPU_UNARY_OPERATOR(gamma, z)
