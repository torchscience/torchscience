// src/torchscience/csrc/autograd/statistics/descriptive/kurtosis.h
#pragma once

#include "../../reduction_macros.h"

// Use the reduction macro with extra parameters (fisher, bias)
// TSCI_EXTRA_2BOOL bundles EXTRA_PARAMS, EXTRA_ARGS, EXTRA_TYPES,
// EXTRA_SAVE, EXTRA_LOAD, and EXTRA_GRAD_PLACEHOLDERS into one macro
TORCHSCIENCE_AUTOGRAD_DIM_REDUCTION_UNARY_OPERATOR_EX(
    statistics::descriptive,
    kurtosis,
    Kurtosis,
    input,
    TSCI_EXTRA_2BOOL(fisher, bias)
)
