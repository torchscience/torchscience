// src/torchscience/csrc/impl/statistics/descriptive/kurtosis_traits.h
#pragma once

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <tuple>

namespace torchscience::impl::statistics::descriptive {

struct KurtosisImpl {
    static constexpr const char* name = "kurtosis";

    // Forward dispatch
    static at::Tensor dispatch_forward(
        const at::Tensor& input,
        at::OptionalIntArrayRef dim,
        bool keepdim,
        bool fisher,
        bool bias
    ) {
        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::kurtosis", "")
            .typed<at::Tensor(
                const at::Tensor&,
                at::OptionalIntArrayRef,
                bool,
                bool,
                bool
            )>()
            .call(input, dim, keepdim, fisher, bias);
    }

    // Backward dispatch
    static at::Tensor dispatch_backward(
        const at::Tensor& grad_output,
        const at::Tensor& input,
        at::OptionalIntArrayRef dim,
        bool keepdim,
        bool fisher,
        bool bias
    ) {
        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::kurtosis_backward", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                at::OptionalIntArrayRef,
                bool,
                bool,
                bool
            )>()
            .call(grad_output, input, dim, keepdim, fisher, bias);
    }

    // Backward-backward dispatch
    static std::tuple<at::Tensor, at::Tensor> dispatch_backward_backward(
        const at::Tensor& grad_grad_input,
        const at::Tensor& grad_output,
        const at::Tensor& input,
        at::OptionalIntArrayRef dim,
        bool keepdim,
        bool fisher,
        bool bias
    ) {
        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::kurtosis_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                at::OptionalIntArrayRef,
                bool,
                bool,
                bool
            )>()
            .call(grad_grad_input, grad_output, input, dim, keepdim, fisher, bias);
    }
};

}  // namespace torchscience::impl::statistics::descriptive
