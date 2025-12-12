#pragma once

#include "torchscience/csrc/autograd/macros.h"

TORCHSCIENCE_BINARY_AUTOGRAD(HankelH1Function, hankel_h_1, nu, x)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(hankel_h_1)
