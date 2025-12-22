#pragma once

#include <torch/library.h>
#include <ATen/autocast_mode.h>

namespace torchscience::autocast::integral_transform {

/**
 * Autocast wrapper for inverse_hilbert_transform.
 * Handles automatic mixed precision casting.
 */
inline at::Tensor inverse_hilbert_transform(
    const at::Tensor& input,
    int64_t n,
    int64_t dim
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    // Determine target dtype based on autocast settings
    auto target_dtype = at::autocast::get_autocast_dtype(
        input.device().is_cpu() ? at::kCPU : at::kCUDA
    );

    // Cast input to target dtype
    at::Tensor input_cast = at::autocast::cached_cast(target_dtype, input);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::inverse_hilbert_transform", "")
        .typed<at::Tensor(const at::Tensor&, int64_t, int64_t)>()
        .call(input_cast, n, dim);
}

}  // namespace torchscience::autocast::integral_transform

TORCH_LIBRARY_IMPL(torchscience, Autocast, module) {
    module.impl(
        "inverse_hilbert_transform",
        &torchscience::autocast::integral_transform::inverse_hilbert_transform
    );
}
