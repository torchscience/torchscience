#pragma once

#include "torchscience/csrc/autograd/macros.h"

TORCHSCIENCE_BINARY_AUTOGRAD(LogBetaFunction, log_beta, a, b)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(log_beta)
