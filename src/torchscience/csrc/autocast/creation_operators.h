#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast {

template<typename CreationTraits>
struct AutocastCreationOperator {
    template<typename... Args>
    static at::Tensor forward(
        Args... args,
        const c10::optional<at::ScalarType>& dtype,
        const c10::optional<at::Layout>& layout,
        const c10::optional<at::Device>& device,
        bool requires_grad
    ) {
        c10::impl::ExcludeDispatchKeyGuard exclude_autocast(
            c10::DispatchKey::Autocast
        );

        // If dtype not specified, use autocast dtype based on device
        c10::optional<at::ScalarType> effective_dtype = dtype;
        if (!dtype.has_value()) {
            auto dev = device.value_or(at::kCPU);
            if (dev.is_cuda()) {
                effective_dtype = at::autocast::get_autocast_dtype(at::kCUDA);
            } else {
                effective_dtype = at::autocast::get_autocast_dtype(at::kCPU);
            }
        }

        return CreationTraits::dispatch_to_backend(
            args...,
            effective_dtype,
            layout,
            device,
            requires_grad
        );
    }
};

#define REGISTER_AUTOCAST_CREATION(module, name, Traits, ...) \
    module.impl(#name, &::torchscience::autocast::AutocastCreationOperator<Traits>::forward<__VA_ARGS__>)

}  // namespace torchscience::autocast
