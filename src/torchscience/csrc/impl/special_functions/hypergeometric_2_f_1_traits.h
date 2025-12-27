#pragma once

/*
 * Hypergeometric2F1Impl - Traits for template-based operator registration
 *
 * This file provides the traits struct that bridges the element-wise
 * hypergeometric 2F1 function implementations to the tensor-level operator
 * templates (CPUQuaternaryOperator, AutogradQuaternaryOperator, etc.).
 */

#include <tuple>

#include <ATen/ATen.h>
#include <c10/macros/Macros.h>

#include "../../core/dispatch_helpers.h"
#include "hypergeometric_2_f_1.h"
#include "hypergeometric_2_f_1_backward.h"
#include "hypergeometric_2_f_1_backward_backward.h"

namespace torchscience::impl::special_functions {

// Declare operator name for template instantiation
DECLARE_OP_NAME(hypergeometric_2_f_1);

/**
 * Traits struct for Gauss hypergeometric function 2F1(a, b; c; z) that provides
 * forward/backward methods compatible with the CPUQuaternaryOperator template.
 */
struct Hypergeometric2F1Impl {
    // Element-wise operations for CPU kernel dispatch
    template<typename T>
    static C10_HOST_DEVICE C10_ALWAYS_INLINE T forward(T a, T b, T c, T z) {
        return hypergeometric_2_f_1(a, b, c, z);
    }

    template<typename T>
    static C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<T, T, T, T> backward(
        T grad, T a, T b, T c, T z
    ) {
        return hypergeometric_2_f_1_backward(grad, a, b, c, z);
    }

    template<typename T>
    static C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<T, T, T, T, T> backward_backward(
        T gg1, T gg2, T gg3, T gg4, T grad, T a, T b, T c, T z,
        bool has_gg1, bool has_gg2, bool has_gg3, bool has_gg4
    ) {
        return hypergeometric_2_f_1_backward_backward(
            gg1, gg2, gg3, gg4, grad, a, b, c, z,
            has_gg1, has_gg2, has_gg3, has_gg4
        );
    }

    // Tensor-level dispatch methods - delegated to dispatch helper
    using Dispatch = ::torchscience::core::QuaternaryDispatch<hypergeometric_2_f_1_op_name>;

    static at::Tensor dispatch_forward(
        const at::Tensor& a,
        const at::Tensor& b,
        const at::Tensor& c,
        const at::Tensor& z
    ) {
        return Dispatch::forward(a, b, c, z);
    }

    static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> dispatch_backward(
        const at::Tensor& grad_output,
        const at::Tensor& a,
        const at::Tensor& b,
        const at::Tensor& c,
        const at::Tensor& z
    ) {
        return Dispatch::backward(grad_output, a, b, c, z);
    }

    static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> dispatch_backward_backward(
        const at::Tensor& grad_grad_a,
        const at::Tensor& grad_grad_b,
        const at::Tensor& grad_grad_c,
        const at::Tensor& grad_grad_z,
        const at::Tensor& grad_output,
        const at::Tensor& a,
        const at::Tensor& b,
        const at::Tensor& c,
        const at::Tensor& z
    ) {
        return Dispatch::backward_backward(
            grad_grad_a, grad_grad_b, grad_grad_c, grad_grad_z,
            grad_output, a, b, c, z
        );
    }
};

}  // namespace torchscience::impl::special_functions
