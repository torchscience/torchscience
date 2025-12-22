#pragma once

#include <torch/library.h>
#include <ATen/autocast_mode.h>

namespace torchscience::autocast::descriptive {

/**
 * Autocast wrapper for kurtosis.
 * Handles automatic mixed precision casting.
 */
inline at::Tensor kurtosis(
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    bool fisher,
    bool bias
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    // Determine target dtype based on autocast settings
    auto target_dtype = at::autocast::get_autocast_dtype(
        input.device().is_cpu() ? at::kCPU : at::kCUDA
    );

    // Cast input to target dtype
    at::Tensor input_cast = at::autocast::cached_cast(target_dtype, input);

    // Store dim as local vector for call
    at::OptionalIntArrayRef dim_ref;
    std::vector<int64_t> dim_vec;
    if (dim.has_value()) {
        dim_vec = std::vector<int64_t>(dim->begin(), dim->end());
        dim_ref = dim_vec;
    }

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::kurtosis", "")
        .typed<at::Tensor(
            const at::Tensor&,
            at::OptionalIntArrayRef,
            bool,
            bool,
            bool
        )>()
        .call(input_cast, dim_ref, keepdim, fisher, bias);
}

}  // namespace torchscience::autocast::descriptive

TORCH_LIBRARY_IMPL(torchscience, Autocast, module) {
    module.impl(
        "kurtosis",
        &torchscience::autocast::descriptive::kurtosis
    );
}
