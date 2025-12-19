#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::cpu::window_function {

inline at::Tensor rectangular_window(
    int64_t m,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Device> device,
    bool requires_grad
) {
    // Handle empty window case
    if (m <= 0) {
        auto options = at::TensorOptions()
            .dtype(dtype.value_or(at::kFloat))
            .device(device.value_or(at::kCPU))
            .requires_grad(requires_grad);
        return at::empty({0}, options);
    }

    // Create tensor of ones
    auto options = at::TensorOptions()
        .dtype(dtype.value_or(at::kFloat))
        .device(device.value_or(at::kCPU))
        .requires_grad(requires_grad);

    return at::ones({m}, options);
}

}  // namespace torchscience::cpu::window_function

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "rectangular_window",
        &torchscience::cpu::window_function::rectangular_window
    );
}
