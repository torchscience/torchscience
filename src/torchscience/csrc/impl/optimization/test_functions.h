#pragma once

#include <ATen/core/Tensor.h>

namespace torchscience::impl::optimization::test_functions {

/**
 * Validates input tensor for Rosenbrock function.
 *
 * Requirements:
 * - Input must be floating-point or complex type
 * - Input must have at least 2 elements in the last dimension
 *
 * @param x Input tensor to validate
 * @param fn_name Name of the calling function for error messages
 * @throws c10::Error if validation fails
 */
inline void check_rosenbrock_input(const at::Tensor& x, const char* fn_name) {
    TORCH_CHECK(
        at::isFloatingType(x.scalar_type()) || at::isComplexType(x.scalar_type()),
        fn_name, " requires floating-point or complex input, got ",
        x.scalar_type()
    );
    TORCH_CHECK(
        x.dim() >= 1 && x.size(-1) >= 2,
        fn_name, " requires at least 2 dimensions in the last axis, got shape ",
        x.sizes()
    );
}

}  // namespace torchscience::impl::optimization::test_functions
