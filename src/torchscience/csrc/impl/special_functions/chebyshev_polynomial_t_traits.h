#pragma once

/*
 * ChebyshevPolynomialTImpl - Traits for template-based operator registration
 *
 * This file provides the traits struct that bridges the element-wise Chebyshev
 * polynomial implementations to the tensor-level operator templates
 * (CPUBinaryOperator, AutogradBinaryOperator, etc.).
 */

#include <tuple>

#include <ATen/ATen.h>
#include <c10/macros/Macros.h>

#include "../../core/dispatch_helpers.h"
#include "chebyshev_polynomial_t.h"

namespace torchscience::impl::special_functions {

// Declare operator name for template instantiation
DECLARE_OP_NAME(chebyshev_polynomial_t);

/**
 * Traits struct for Chebyshev polynomial T_v(z) that provides forward/backward
 * methods compatible with the CPUBinaryOperator template.
 */
struct ChebyshevPolynomialTImpl {
    // Element-wise operations for CPU kernel dispatch
    template<typename T>
    static C10_HOST_DEVICE C10_ALWAYS_INLINE T forward(T v, T z) {
        return chebyshev_polynomial_t(v, z);
    }

    template<typename T>
    static C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<T, T> backward(
        T grad, T v, T z
    ) {
        return chebyshev_polynomial_t_backward(grad, v, z);
    }

    template<typename T>
    static C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<T, T, T> backward_backward(
        T gg1, T gg2, T grad, T v, T z, bool has_gg1, bool has_gg2
    ) {
        return chebyshev_polynomial_t_backward_backward(gg1, gg2, grad, v, z, has_gg1, has_gg2);
    }

    // Tensor-level dispatch methods - delegated to dispatch helper
    using Dispatch = ::torchscience::core::BinaryDispatch<chebyshev_polynomial_t_op_name>;

    static at::Tensor dispatch_forward(
        const at::Tensor& v,
        const at::Tensor& z
    ) {
        return Dispatch::forward(v, z);
    }

    static std::tuple<at::Tensor, at::Tensor> dispatch_backward(
        const at::Tensor& grad_output,
        const at::Tensor& v,
        const at::Tensor& z
    ) {
        return Dispatch::backward(grad_output, v, z);
    }

    static std::tuple<at::Tensor, at::Tensor, at::Tensor> dispatch_backward_backward(
        const at::Tensor& grad_grad_v,
        const at::Tensor& grad_grad_z,
        const at::Tensor& grad_output,
        const at::Tensor& v,
        const at::Tensor& z
    ) {
        return Dispatch::backward_backward(grad_grad_v, grad_grad_z, grad_output, v, z);
    }
};

}  // namespace torchscience::impl::special_functions
