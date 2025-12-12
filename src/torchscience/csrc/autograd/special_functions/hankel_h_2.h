#pragma once

#include "torchscience/csrc/autograd/macros.h"

TORCHSCIENCE_BINARY_AUTOGRAD(HankelH2Function, hankel_h_2, nu, x)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(hankel_h_2)
