#pragma once

#include <vector>

#include <torch/extension.h>

// =============================================================================
// HELPER MACROS
// =============================================================================

// Use these to wrap extra parameters for _EX macros
// TSCI_EXTRA(bool fisher, bool bias) expands to: , bool fisher, bool bias
// TSCI_TYPES(bool, bool) expands to: , bool, bool
#ifndef TSCI_EXTRA
#define TSCI_EXTRA(...) , __VA_ARGS__
#endif
#ifndef TSCI_NO_EXTRA
#define TSCI_NO_EXTRA
#endif
#define TSCI_TYPES(...) , __VA_ARGS__
#define TSCI_NO_TYPES

// Macros for saving/loading extra params in autograd context
// Usage: TSCI_SAVE(ctx->saved_data["fisher"] = fisher; ctx->saved_data["bias"] = bias;)
// Usage: TSCI_LOAD(bool fisher = ctx->saved_data["fisher"].toBool(); bool bias = ctx->saved_data["bias"].toBool();)
#define TSCI_SAVE(...) __VA_ARGS__
#define TSCI_LOAD(...) __VA_ARGS__
#define TSCI_NO_SAVE
#define TSCI_NO_LOAD

// Placeholder returns for extra params in backward (one at::Tensor() per extra param)
// Usage: TSCI_GRAD_PLACEHOLDERS(at::Tensor(), at::Tensor()) for 2 extra params
#define TSCI_GRAD_PLACEHOLDERS(...) , __VA_ARGS__
#define TSCI_NO_GRAD_PLACEHOLDERS

// =============================================================================
// CONVENIENCE MACROS FOR COMMON EXTRA PARAMETER PATTERNS
// =============================================================================
//
// These macros reduce the parameter count when using _EX macros by bundling
// related declarations together. Instead of 10 separate parameters, you can
// use 5 parameters with these helpers.
//
// Example - Before (10 params):
//   TORCHSCIENCE_AUTOGRAD_DIM_REDUCTION_UNARY_OPERATOR_EX(
//       statistics::descriptive, kurtosis, Kurtosis, input,
//       TSCI_EXTRA(bool fisher, bool bias),
//       TSCI_EXTRA(fisher, bias),
//       TSCI_TYPES(bool, bool),
//       TSCI_SAVE(ctx->saved_data["fisher"] = fisher; ctx->saved_data["bias"] = bias;),
//       TSCI_LOAD(bool fisher = ctx->saved_data["fisher"].toBool(); ...),
//       TSCI_GRAD_PLACEHOLDERS(at::Tensor(), at::Tensor())
//   )
//
// Example - After (5 params):
//   TORCHSCIENCE_AUTOGRAD_DIM_REDUCTION_UNARY_OPERATOR_EX(
//       statistics::descriptive, kurtosis, Kurtosis, input,
//       TSCI_EXTRA_2BOOL(fisher, bias)
//   )

// Individual parameter save/load helpers
#define TSCI_BOOL_SAVE(n) ctx->saved_data[#n] = n;
#define TSCI_BOOL_LOAD(n) bool n = ctx->saved_data[#n].toBool();
#define TSCI_INT_SAVE(n) ctx->saved_data[#n] = n;
#define TSCI_INT_LOAD(n) int64_t n = ctx->saved_data[#n].toInt();
#define TSCI_DOUBLE_SAVE(n) ctx->saved_data[#n] = n;
#define TSCI_DOUBLE_LOAD(n) double n = ctx->saved_data[#n].toDouble();

// Combined helper for 1 bool parameter
// Expands to: EXTRA_PARAMS, EXTRA_ARGS, EXTRA_TYPES, EXTRA_SAVE, EXTRA_LOAD, EXTRA_GRAD_PLACEHOLDERS
#define TSCI_EXTRA_1BOOL(n1) \
    TSCI_EXTRA(bool n1), \
    TSCI_EXTRA(n1), \
    TSCI_TYPES(bool), \
    TSCI_SAVE(TSCI_BOOL_SAVE(n1)), \
    TSCI_LOAD(TSCI_BOOL_LOAD(n1)), \
    TSCI_GRAD_PLACEHOLDERS(at::Tensor())

// Combined helper for 2 bool parameters
#define TSCI_EXTRA_2BOOL(n1, n2) \
    TSCI_EXTRA(bool n1, bool n2), \
    TSCI_EXTRA(n1, n2), \
    TSCI_TYPES(bool, bool), \
    TSCI_SAVE(TSCI_BOOL_SAVE(n1) TSCI_BOOL_SAVE(n2)), \
    TSCI_LOAD(TSCI_BOOL_LOAD(n1) TSCI_BOOL_LOAD(n2)), \
    TSCI_GRAD_PLACEHOLDERS(at::Tensor(), at::Tensor())

// Combined helper for 3 bool parameters
#define TSCI_EXTRA_3BOOL(n1, n2, n3) \
    TSCI_EXTRA(bool n1, bool n2, bool n3), \
    TSCI_EXTRA(n1, n2, n3), \
    TSCI_TYPES(bool, bool, bool), \
    TSCI_SAVE(TSCI_BOOL_SAVE(n1) TSCI_BOOL_SAVE(n2) TSCI_BOOL_SAVE(n3)), \
    TSCI_LOAD(TSCI_BOOL_LOAD(n1) TSCI_BOOL_LOAD(n2) TSCI_BOOL_LOAD(n3)), \
    TSCI_GRAD_PLACEHOLDERS(at::Tensor(), at::Tensor(), at::Tensor())

// Combined helper for 1 int64_t parameter
#define TSCI_EXTRA_1INT(n1) \
    TSCI_EXTRA(int64_t n1), \
    TSCI_EXTRA(n1), \
    TSCI_TYPES(int64_t), \
    TSCI_SAVE(TSCI_INT_SAVE(n1)), \
    TSCI_LOAD(TSCI_INT_LOAD(n1)), \
    TSCI_GRAD_PLACEHOLDERS(at::Tensor())

// Combined helper for 1 double parameter
#define TSCI_EXTRA_1DOUBLE(n1) \
    TSCI_EXTRA(double n1), \
    TSCI_EXTRA(n1), \
    TSCI_TYPES(double), \
    TSCI_SAVE(TSCI_DOUBLE_SAVE(n1)), \
    TSCI_LOAD(TSCI_DOUBLE_LOAD(n1)), \
    TSCI_GRAD_PLACEHOLDERS(at::Tensor())

// Combined helper for 1 bool + 1 int64_t parameter
#define TSCI_EXTRA_BOOL_INT(n1, n2) \
    TSCI_EXTRA(bool n1, int64_t n2), \
    TSCI_EXTRA(n1, n2), \
    TSCI_TYPES(bool, int64_t), \
    TSCI_SAVE(TSCI_BOOL_SAVE(n1) TSCI_INT_SAVE(n2)), \
    TSCI_LOAD(TSCI_BOOL_LOAD(n1) TSCI_INT_LOAD(n2)), \
    TSCI_GRAD_PLACEHOLDERS(at::Tensor(), at::Tensor())

// =============================================================================
// DIM-BASED REDUCTION MACROS (Autograd)
// =============================================================================

/**
 * Autograd macro for unary dim-based reduction operators (no extra params).
 */
