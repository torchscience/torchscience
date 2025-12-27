#pragma once

/*
 * GammaImpl - Traits for template-based operator registration
 *
 * This file provides the traits struct that bridges the element-wise gamma
 * implementations to the tensor-level operator templates (CPUUnaryOperator,
 * AutogradUnaryOperator, etc.).
 */

#include <tuple>

#include <ATen/ATen.h>
#include <c10/macros/Macros.h>

#include "../../core/dispatch_helpers.h"
#include "gamma.h"
#include "gamma_backward.h"
#include "gamma_backward_backward.h"

namespace torchscience::impl::special_functions {

// Declare operator name for template instantiation
DECLARE_OP_NAME(gamma);

/**
 * Traits struct for gamma function that provides forward/backward methods
 * compatible with the CPUUnaryOperator template.
 */
struct GammaImpl {
    // Element-wise operations for CPU kernel dispatch
    template<typename T>
    static C10_HOST_DEVICE C10_ALWAYS_INLINE T forward(T z) {
        return gamma(z);
    }

    template<typename T>
    static C10_HOST_DEVICE C10_ALWAYS_INLINE T backward(T grad, T z) {
        return gamma_backward(grad, z);
    }

    template<typename T>
    static C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<T, T> backward_backward(
        T gg, T grad, T z, bool has_gg
    ) {
        return gamma_backward_backward(gg, grad, z, has_gg);
    }

    // Tensor-level dispatch methods - delegated to dispatch helper
    using Dispatch = ::torchscience::core::UnaryDispatch<gamma_op_name>;

    static at::Tensor dispatch_forward(const at::Tensor& input) {
        return Dispatch::forward(input);
    }

    static at::Tensor dispatch_backward(
        const at::Tensor& grad_output,
        const at::Tensor& input
    ) {
        return Dispatch::backward(grad_output, input);
    }

    static std::tuple<at::Tensor, at::Tensor> dispatch_backward_backward(
        const at::Tensor& grad_grad_input,
        const at::Tensor& grad_output,
        const at::Tensor& input
    ) {
        return Dispatch::backward_backward(grad_grad_input, grad_output, input);
    }
};

}  // namespace torchscience::impl::special_functions
