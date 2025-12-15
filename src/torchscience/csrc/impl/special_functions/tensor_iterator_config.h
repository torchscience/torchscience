#pragma once

#include <ATen/TensorIterator.h>

namespace torchscience {
namespace impl {
namespace special_functions {

/**
 * Common TensorIterator configuration for binary special functions.
 *
 * This utility reduces code duplication between CPU and CUDA implementations
 * by providing a consistent configuration for element-wise operations.
 */
inline at::TensorIterator make_binary_iterator(
    at::Tensor& output,
    const at::Tensor& input1,
    const at::Tensor& input2
) {
  return at::TensorIteratorConfig()
    .add_output(output)
    .add_const_input(input1)
    .add_const_input(input2)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .enforce_safe_casting_to_output(false)
    .build();
}

/**
 * Common TensorIterator configuration for ternary special functions
 * (e.g., backward passes with grad_output, v, z).
 */
inline at::TensorIterator make_ternary_iterator(
    at::Tensor& output,
    const at::Tensor& input1,
    const at::Tensor& input2,
    const at::Tensor& input3
) {
  return at::TensorIteratorConfig()
    .add_output(output)
    .add_const_input(input1)
    .add_const_input(input2)
    .add_const_input(input3)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .enforce_safe_casting_to_output(false)
    .build();
}

/**
 * Common TensorIterator configuration for binary special functions with
 * pre-allocated output tensor.
 *
 * When output is already allocated (out= parameter), this ensures the
 * computation writes directly to the provided tensor.
 */
inline at::TensorIterator make_binary_iterator_with_output(
    at::Tensor& output,
    const at::Tensor& input1,
    const at::Tensor& input2
) {
  return at::TensorIteratorConfig()
    .add_output(output)
    .add_const_input(input1)
    .add_const_input(input2)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .enforce_safe_casting_to_output(false)
    .resize_outputs(false)  // Don't resize, output is pre-allocated
    .build();
}

/**
 * Common TensorIterator configuration for ternary special functions with
 * pre-allocated output tensor.
 */
inline at::TensorIterator make_ternary_iterator_with_output(
    at::Tensor& output,
    const at::Tensor& input1,
    const at::Tensor& input2,
    const at::Tensor& input3
) {
  return at::TensorIteratorConfig()
    .add_output(output)
    .add_const_input(input1)
    .add_const_input(input2)
    .add_const_input(input3)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .enforce_safe_casting_to_output(false)
    .resize_outputs(false)  // Don't resize, output is pre-allocated
    .build();
}

/**
 * Common TensorIterator configuration for ternary functions with dual outputs.
 * Used for fused backward passes that compute two gradients in a single pass.
 */
inline at::TensorIterator make_ternary_dual_output_iterator(
    at::Tensor& output1,
    at::Tensor& output2,
    const at::Tensor& input1,
    const at::Tensor& input2,
    const at::Tensor& input3
) {
  return at::TensorIteratorConfig()
    .add_output(output1)
    .add_output(output2)
    .add_const_input(input1)
    .add_const_input(input2)
    .add_const_input(input3)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .enforce_safe_casting_to_output(false)
    .build();
}

} // namespace special_functions
} // namespace impl
} // namespace torchscience
