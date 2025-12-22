#pragma once

#include <limits>
#include <vector>
#include <ATen/ATen.h>

namespace torchscience::core {

// Validate shape parameters
inline void check_size_nonnegative(const std::vector<int64_t>& shape, const char* op_name) {
    for (auto s : shape) {
        TORCH_CHECK(s >= 0, op_name, ": size must be non-negative, got ", s);
    }
}

// Compute total number of elements from shape (with overflow checking)
inline int64_t compute_numel(const std::vector<int64_t>& shape) {
    int64_t numel = 1;
    for (auto s : shape) {
        // Check for overflow before multiplication
        TORCH_CHECK(
            numel <= std::numeric_limits<int64_t>::max() / (s == 0 ? 1 : s),
            "numel would overflow int64_t for shape with total elements exceeding 2^63-1"
        );
        numel *= s;
    }
    return numel;
}

// Build TensorOptions from optional parameters
inline at::TensorOptions build_options(
    const c10::optional<at::ScalarType>& dtype,
    const c10::optional<at::Layout>& layout,
    const c10::optional<at::Device>& device,
    at::Device default_device = at::kCPU
) {
    return at::TensorOptions()
        .dtype(dtype.value_or(c10::typeMetaToScalarType(at::get_default_dtype())))
        .layout(layout.value_or(at::kStrided))
        .device(device.value_or(default_device))
        .requires_grad(false);
}

} // namespace torchscience::core
