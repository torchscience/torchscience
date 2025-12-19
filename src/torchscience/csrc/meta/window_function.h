#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::window_function {

inline at::Tensor rectangular_window(
    int64_t m,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Device> device,
    bool requires_grad
) {
    // For meta tensors, we just return a tensor with the correct shape
    // without actually allocating memory
    int64_t size = m > 0 ? m : 0;
    auto options = at::TensorOptions()
        .dtype(dtype.value_or(at::kFloat))
        .device(at::kMeta);

    return at::empty({size}, options);
}

}  // namespace torchscience::meta::window_function

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl(
        "rectangular_window",
        &torchscience::meta::window_function::rectangular_window
    );
}
