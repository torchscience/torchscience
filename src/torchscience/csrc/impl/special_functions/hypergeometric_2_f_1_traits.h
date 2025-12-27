#pragma once

#include <tuple>

#include <c10/macros/Macros.h>

#include "../../core/dispatch_helpers.h"
#include "hypergeometric_2_f_1.h"
#include "hypergeometric_2_f_1_backward.h"
#include "hypergeometric_2_f_1_backward_backward.h"

namespace torchscience::impl::special_functions {

DECLARE_OP_NAME(hypergeometric_2_f_1);

struct Hypergeometric2F1Impl {
    template<typename T>
    static C10_HOST_DEVICE C10_ALWAYS_INLINE T forward(
        T a,
        T b,
        T c,
        T z
    ) {
        return hypergeometric_2_f_1(
            a,
            b,
            c,
            z
        );
    }

    template<typename T>
    static C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<
        T,
        T,
        T,
        T
    > backward(
        T gradient,
        T a,
        T b,
        T c,
        T z
    ) {
        return hypergeometric_2_f_1_backward(
            gradient,
            a,
            b,
            c,
            z
        );
    }

    template<typename T>
    static C10_HOST_DEVICE C10_ALWAYS_INLINE std::tuple<
        T,
        T,
        T,
        T,
        T
    > backward_backward(
        T gradient_gradient_a,
        T gradient_gradient_b,
        T gradient_gradient_c,
        T gradient_gradient_z,
        T gradient,
        T a,
        T b,
        T c,
        T z,
        bool has_gradient_gradient_a,
        bool has_gradient_gradient_b,
        bool has_gradient_gradient_c,
        bool has_gradient_gradient_z
    ) {
        return hypergeometric_2_f_1_backward_backward(
            gradient_gradient_a,
            gradient_gradient_b,
            gradient_gradient_c,
            gradient_gradient_z,
            gradient,
            a,
            b,
            c,
            z,
            has_gradient_gradient_a,
            has_gradient_gradient_b,
            has_gradient_gradient_c,
            has_gradient_gradient_z
        );
    }

    using Dispatch = core::QuaternaryDispatch<hypergeometric_2_f_1_op_name>;

    static at::Tensor dispatch_forward(
        const at::Tensor& a,
        const at::Tensor& b,
        const at::Tensor& c,
        const at::Tensor& z
    ) {
        return Dispatch::forward(
            a,
            b,
            c,
            z
        );
    }

    static std::tuple<
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor
    > dispatch_backward(
        const at::Tensor& gradient_output,
        const at::Tensor& a,
        const at::Tensor& b,
        const at::Tensor& c,
        const at::Tensor& z
    ) {
        return Dispatch::backward(
            gradient_output,
            a,
            b,
            c,
            z
        );
    }

    static std::tuple<
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor
    > dispatch_backward_backward(
        const at::Tensor& gradient_gradient_a,
        const at::Tensor& gradient_gradient_b,
        const at::Tensor& gradient_gradient_c,
        const at::Tensor& gradient_gradient_z,
        const at::Tensor& gradient_output,
        const at::Tensor& a,
        const at::Tensor& b,
        const at::Tensor& c,
        const at::Tensor& z
    ) {
        return Dispatch::backward_backward(
            gradient_gradient_a,
            gradient_gradient_b,
            gradient_gradient_c,
            gradient_gradient_z,
            gradient_output,
            a,
            b,
            c,
            z
        );
    }
};

}  // namespace torchscience::impl::special_functions
