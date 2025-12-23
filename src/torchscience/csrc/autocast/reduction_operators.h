#pragma once

#include <ATen/autocast_mode.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::autocast {

// =============================================================================
// AutocastReductionOperator - Dtype casting for reduction operators
// =============================================================================

struct AutocastReductionOperator {
    template<typename Dispatcher, typename... Args>
    static at::Tensor forward(
        const at::Tensor& input,
        at::OptionalIntArrayRef dim,
        bool keepdim,
        Args... args
    ) {
        c10::impl::ExcludeDispatchKeyGuard exclude_autocast(
            c10::DispatchKey::Autocast
        );

        at::ScalarType target_dtype = at::autocast::get_autocast_dtype(
            at::kCPU
        );

        return Dispatcher::dispatch_forward(
            at::autocast::cached_cast(target_dtype, input),
            dim,
            keepdim,
            args...
        );
    }

    template<typename Dispatcher, typename... Args>
    static at::Tensor backward(
        const at::Tensor& grad_output,
        const at::Tensor& input,
        at::OptionalIntArrayRef dim,
        bool keepdim,
        Args... args
    ) {
        c10::impl::ExcludeDispatchKeyGuard exclude_autocast(
            c10::DispatchKey::Autocast
        );

        at::ScalarType target_dtype = at::autocast::get_autocast_dtype(
            at::kCPU
        );

        return Dispatcher::dispatch_backward(
            at::autocast::cached_cast(target_dtype, grad_output),
            at::autocast::cached_cast(target_dtype, input),
            dim,
            keepdim,
            args...
        );
    }
};

}  // namespace torchscience::autocast
