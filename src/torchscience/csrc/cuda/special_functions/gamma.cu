#define TORCH_ASSERT_NO_OPERATORS

#include <tuple>

#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/TensorIterator.h>

#include "../../impl/special_functions/gamma.h"
#include "../../impl/special_functions/tensor_iterator_config.h"

namespace torchscience::cuda::special_functions {

at::Tensor gamma(const at::Tensor& z) {
  at::Tensor output;
  auto iterator = impl::special_functions::make_unary_iterator(output, z);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::kBFloat16, at::kHalf,
      iterator.common_dtype(),
      "gamma_cuda",
      [&]() {
        at::native::gpu_kernel(iterator, []GPU_LAMBDA(scalar_t z) -> scalar_t {
          return impl::special_functions::gamma<scalar_t>(z);
        });
      });

  return iterator.output();
}

at::Tensor gamma_backward(
    const at::Tensor& grad_output,
    const at::Tensor& z
) {
  at::Tensor gradient_z;
  auto iterator = impl::special_functions::make_binary_iterator(
      gradient_z, grad_output, z);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::kBFloat16, at::kHalf,
      iterator.common_dtype(),
      "gamma_backward_cuda",
      [&]() {
        at::native::gpu_kernel(
            iterator,
            []GPU_LAMBDA(scalar_t grad_output, scalar_t z) -> scalar_t {
              return impl::special_functions::gamma_backward<scalar_t>(
                  grad_output, z);
            });
      });

  return iterator.output();
}

std::tuple<at::Tensor, at::Tensor> gamma_backward_backward(
    const at::Tensor& gg_z,
    const at::Tensor& grad_output,
    const at::Tensor& z
) {
  const bool has_gg_z = gg_z.defined();

  if (!has_gg_z) {
    return std::make_tuple(at::Tensor(), at::Tensor());
  }

  at::Tensor gradient_grad_output;
  at::Tensor gradient_z;

  auto iterator = impl::special_functions::make_ternary_dual_output_iterator(
      gradient_grad_output, gradient_z, gg_z, grad_output, z);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::kBFloat16, at::kHalf,
      iterator.common_dtype(),
      "gamma_backward_backward_cuda",
      [&]() {
        at::native::gpu_kernel_multiple_outputs(
            iterator,
            []GPU_LAMBDA(scalar_t gg_z, scalar_t grad_output, scalar_t z)
                -> thrust::tuple<scalar_t, scalar_t> {
              auto result = impl::special_functions::gamma_backward_backward<scalar_t>(
                  gg_z, grad_output, z, true);
              return thrust::make_tuple(
                  std::get<0>(result),
                  std::get<1>(result));
            });
      });

  return std::make_tuple(iterator.output(0), iterator.output(1));
}

}  // namespace torchscience::cuda::special_functions
