#pragma once

/*
 * IncompleteBetaImpl - Traits for template-based operator registration
 *
 * This file provides the traits struct that bridges the element-wise incomplete
 * beta function implementations to the tensor-level operator templates
 * (CPUTernaryOperator, AutogradTernaryOperator, etc.).
 */

#include <tuple>

#include <ATen/ATen.h>
#include <c10/macros/Macros.h>

#include "../../core/dispatch_helpers.h"
#include "incomplete_beta.h"
#include "incomplete_beta_backward.h"
#include "incomplete_beta_backward_backward.h"

namespace torchscience::impl::special_functions {

// Declare operator name for template instantiation
DECLARE_OP_NAME(incomplete_beta);

/**
 * Traits struct for regularized incomplete beta function I_z(a, b) that provides
 * forward/backward methods compatible with the CPUTernaryOperator template.
 */
struct IncompleteBetaImpl {
    // Element-wise operations for CPU kernel dispatch
    template<typename T>
    static C10_HOST_DEVICE C10_ALWAYS_INLINE T forward(T z, T a, T b) {
        return incomplete_beta(z, a, b);
    }

    template<typename T>
    static C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<T, T, T> backward(
        T grad, T z, T a, T b
    ) {
        return incomplete_beta_backward(grad, z, a, b);
    }

    template<typename T>
    static C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<T, T, T, T> backward_backward(
        T gg1, T gg2, T gg3, T grad, T z, T a, T b,
        bool has_gg1, bool has_gg2, bool has_gg3
    ) {
        return incomplete_beta_backward_backward(gg1, gg2, gg3, grad, z, a, b, has_gg1, has_gg2, has_gg3);
    }

    // Tensor-level dispatch methods - delegated to dispatch helper
    using Dispatch = ::torchscience::core::TernaryDispatch<incomplete_beta_op_name>;

    static at::Tensor dispatch_forward(
        const at::Tensor& z,
        const at::Tensor& a,
        const at::Tensor& b
    ) {
        return Dispatch::forward(z, a, b);
    }

    static std::tuple<at::Tensor, at::Tensor, at::Tensor> dispatch_backward(
        const at::Tensor& grad_output,
        const at::Tensor& z,
        const at::Tensor& a,
        const at::Tensor& b
    ) {
        return Dispatch::backward(grad_output, z, a, b);
    }

    static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> dispatch_backward_backward(
        const at::Tensor& grad_grad_z,
        const at::Tensor& grad_grad_a,
        const at::Tensor& grad_grad_b,
        const at::Tensor& grad_output,
        const at::Tensor& z,
        const at::Tensor& a,
        const at::Tensor& b
    ) {
        return Dispatch::backward_backward(grad_grad_z, grad_grad_a, grad_grad_b, grad_output, z, a, b);
    }
};

}  // namespace torchscience::impl::special_functions
