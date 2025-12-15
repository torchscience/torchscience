#pragma once

#include <ATen/ATen.h>

namespace torchscience {
namespace quantized {
namespace cuda {
namespace special_functions {

/**
 * Quantized CUDA implementation for chebyshev_polynomial_t.
 *
 * This implementation dequantizes inputs to floating-point, computes the
 * Chebyshev polynomial using the standard CUDA floating-point implementation,
 * and requantizes the result.
 *
 * QUANTIZATION BEHAVIOR:
 *   - Input tensors are dequantized using their stored scale and zero_point
 *   - Computation is performed in floating-point on GPU
 *   - Output is requantized using the quantization parameters from z
 *   - This approach preserves full precision during computation
 *
 * LIMITATIONS:
 *   - Dequantization/requantization overhead may reduce performance benefits
 *     of quantization for compute-bound workloads
 *   - Requantization may introduce additional quantization error
 *   - Output quantization parameters are inherited from z, which may not be
 *     optimal for the output range of T_v(z)
 *
 * For optimal quantized performance, consider:
 *   - Pre-computing quantization parameters for the expected output range
 *   - Using higher precision (e.g., int16) for intermediate values if available
 */
at::Tensor chebyshev_polynomial_t(const at::Tensor& v, const at::Tensor& z) {
  auto v_fp = v.dequantize();
  auto z_fp = z.dequantize();

  at::AutoDispatchBelowAutograd guard;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("torchscience::chebyshev_polynomial_t", "")
      .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>();
  auto result_fp = op.call(v_fp, z_fp);

  return at::quantize_per_tensor(
      result_fp,
      z.q_scale(),
      z.q_zero_point(),
      z.scalar_type());
}

/**
 * Quantized backward for z.
 *
 * Uses z's quantization parameters for the output gradient tensor.
 */
at::Tensor chebyshev_polynomial_t_backward_z(
    const at::Tensor& grad_output,
    const at::Tensor& v,
    const at::Tensor& z
) {
  auto grad_fp = grad_output.dequantize();
  auto v_fp = v.dequantize();
  auto z_fp = z.dequantize();

  at::AutoDispatchBelowAutograd guard;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("torchscience::chebyshev_polynomial_t_backward_z", "")
      .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>();
  auto result_fp = op.call(grad_fp, v_fp, z_fp);

  return at::quantize_per_tensor(
      result_fp,
      z.q_scale(),
      z.q_zero_point(),
      z.scalar_type());
}

/**
 * Quantized backward for v.
 *
 * Uses v's quantization parameters for the output gradient tensor.
 */
at::Tensor chebyshev_polynomial_t_backward_v(
    const at::Tensor& grad_output,
    const at::Tensor& v,
    const at::Tensor& z
) {
  auto grad_fp = grad_output.dequantize();
  auto v_fp = v.dequantize();
  auto z_fp = z.dequantize();

  at::AutoDispatchBelowAutograd guard;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("torchscience::chebyshev_polynomial_t_backward_v", "")
      .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>();
  auto result_fp = op.call(grad_fp, v_fp, z_fp);

  return at::quantize_per_tensor(
      result_fp,
      v.q_scale(),
      v.q_zero_point(),
      v.scalar_type());
}

} // namespace special_functions
} // namespace cuda
} // namespace quantized
} // namespace torchscience
