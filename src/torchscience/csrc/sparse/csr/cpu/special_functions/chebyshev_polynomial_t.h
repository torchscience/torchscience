#pragma once

#include <ATen/ATen.h>

namespace torchscience {
namespace sparse {
namespace csr {
namespace cpu {
namespace special_functions {

/**
 * Sparse CSR implementation for chebyshev_polynomial_t.
 *
 * LIMITATIONS:
 * This implementation materializes sparse tensors to dense, computes the result,
 * and converts back to sparse. This approach:
 *   - Does NOT preserve memory efficiency for large sparse tensors
 *   - Has O(n) memory complexity where n is the dense size, not nnz
 *   - May be slow for very sparse tensors due to materialization overhead
 *
 * This behavior is mathematically necessary because Chebyshev polynomials do not
 * preserve sparsity: T_n(0) = cos(n*pi/2), which is non-zero for most n.
 * A sparse input with zeros will generally produce a dense output.
 *
 * For performance-critical sparse workloads, consider operating on the non-zero
 * values directly if the sparsity pattern permits.
 */
at::Tensor chebyshev_polynomial_t(const at::Tensor& v, const at::Tensor& z) {
  auto v_dense = v.to_dense();
  auto z_dense = z.to_dense();

  at::AutoDispatchBelowAutograd guard;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("torchscience::chebyshev_polynomial_t", "")
      .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>();
  auto result = op.call(v_dense, z_dense);

  return result.to_sparse_csr();
}

at::Tensor chebyshev_polynomial_t_backward_z(
    const at::Tensor& grad_output,
    const at::Tensor& v,
    const at::Tensor& z
) {
  auto grad_dense = grad_output.to_dense();
  auto v_dense = v.to_dense();
  auto z_dense = z.to_dense();

  at::AutoDispatchBelowAutograd guard;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("torchscience::chebyshev_polynomial_t_backward_z", "")
      .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>();
  auto result = op.call(grad_dense, v_dense, z_dense);

  return result.to_sparse_csr();
}

at::Tensor chebyshev_polynomial_t_backward_v(
    const at::Tensor& grad_output,
    const at::Tensor& v,
    const at::Tensor& z
) {
  auto grad_dense = grad_output.to_dense();
  auto v_dense = v.to_dense();
  auto z_dense = z.to_dense();

  at::AutoDispatchBelowAutograd guard;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("torchscience::chebyshev_polynomial_t_backward_v", "")
      .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>();
  auto result = op.call(grad_dense, v_dense, z_dense);

  return result.to_sparse_csr();
}

} // namespace special_functions
} // namespace cpu
} // namespace csr
} // namespace sparse
} // namespace torchscience
