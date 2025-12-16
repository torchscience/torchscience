#define TORCH_ASSERT_NO_OPERATORS

#include <tuple>

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/TensorIterator.h>
#include <c10/cuda/CUDAGuard.h>

#include "../../impl/special_functions/chebyshev_polynomial_t.h"
#include "../../impl/special_functions/tensor_iterator_config.h"

namespace torchscience::cuda::special_functions {

at::Tensor chebyshev_polynomial_t(const at::Tensor& v, const at::Tensor& z) {
  at::Tensor output;
  auto iterator = impl::special_functions::make_binary_iterator(output, v, z);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::kBFloat16, at::kHalf,
      iterator.common_dtype(),
      "chebyshev_polynomial_t_cuda",
      [&]() {
        at::native::gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t v, scalar_t z) -> scalar_t {
          return impl::special_functions::chebyshev_polynomial_t<scalar_t>(v, z);
        });
      });
  return iterator.output();
}

/**
 * CUDA kernel for fused backward computation.
 */
template <typename scalar_t>
__global__ void chebyshev_polynomial_t_backward_kernel(
    const scalar_t* __restrict__ grad,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ z,
    scalar_t* __restrict__ grad_v,
    scalar_t* __restrict__ grad_z,
    int64_t numel,
    bool v_requires_grad
) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    auto result = impl::special_functions::chebyshev_polynomial_t_backward<scalar_t>(
        grad[idx], v[idx], z[idx]);
    grad_v[idx] = v_requires_grad ? std::get<0>(result) : scalar_t(0);
    grad_z[idx] = std::get<1>(result);
  }
}

std::tuple<at::Tensor, at::Tensor> chebyshev_polynomial_t_backward(
    const at::Tensor& grad_output,
    const at::Tensor& v,
    const at::Tensor& z,
    bool v_requires_grad
) {
  auto output_size = at::infer_size(at::infer_size(grad_output.sizes(), v.sizes()), z.sizes());

  auto promoted_dtype = at::promote_types(
      at::promote_types(grad_output.scalar_type(), v.scalar_type()),
      z.scalar_type());

  auto grad_expanded = grad_output.expand(output_size).to(promoted_dtype).contiguous();
  auto v_expanded = v.expand(output_size).to(promoted_dtype).contiguous();
  auto z_expanded = z.expand(output_size).to(promoted_dtype).contiguous();

  auto grad_v_out = at::empty(output_size, grad_output.options().dtype(promoted_dtype));
  auto grad_z_out = at::empty(output_size, grad_output.options().dtype(promoted_dtype));

  int64_t numel = grad_expanded.numel();

  if (numel == 0) {
    return std::make_tuple(grad_v_out, grad_z_out);
  }

  const int threads = 256;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::kBFloat16, at::kHalf,
      promoted_dtype,
      "chebyshev_polynomial_t_backward_cuda",
      [&]() {
        chebyshev_polynomial_t_backward_kernel<scalar_t><<<blocks, threads, 0,
            at::cuda::getCurrentCUDAStream()>>>(
            grad_expanded.data_ptr<scalar_t>(),
            v_expanded.data_ptr<scalar_t>(),
            z_expanded.data_ptr<scalar_t>(),
            grad_v_out.data_ptr<scalar_t>(),
            grad_z_out.data_ptr<scalar_t>(),
            numel,
            v_requires_grad);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return std::make_tuple(grad_v_out, grad_z_out);
}

/**
 * CUDA kernel for fused double-backward computation.
 */
template <typename scalar_t>
__global__ void chebyshev_polynomial_t_backward_backward_kernel(
    const scalar_t* __restrict__ ggv,
    const scalar_t* __restrict__ ggz,
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ z,
    scalar_t* __restrict__ grad_grad_output,
    scalar_t* __restrict__ grad_v,
    scalar_t* __restrict__ grad_z,
    int64_t numel,
    bool has_ggv,
    bool has_ggz
) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    scalar_t ggv_val = has_ggv ? ggv[idx] : scalar_t(0);
    scalar_t ggz_val = has_ggz ? ggz[idx] : scalar_t(0);

    auto result = impl::special_functions::chebyshev_polynomial_t_backward_backward<scalar_t>(
        ggv_val, ggz_val, grad_output[idx], v[idx], z[idx], has_ggv, has_ggz);

    grad_grad_output[idx] = std::get<0>(result);
    grad_v[idx] = std::get<1>(result);
    grad_z[idx] = std::get<2>(result);
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> chebyshev_polynomial_t_backward_backward(
    const at::Tensor& ggv,
    const at::Tensor& ggz,
    const at::Tensor& grad_output,
    const at::Tensor& v,
    const at::Tensor& z,
    bool has_ggv,
    bool has_ggz
) {
  // Determine output size from all inputs
  auto output_size = at::infer_size(grad_output.sizes(), at::infer_size(v.sizes(), z.sizes()));
  if (has_ggv) {
    output_size = at::infer_size(output_size, ggv.sizes());
  }
  if (has_ggz) {
    output_size = at::infer_size(output_size, ggz.sizes());
  }

  // Determine output dtype
  auto promoted_dtype = at::promote_types(
      at::promote_types(grad_output.scalar_type(), v.scalar_type()),
      z.scalar_type());
  if (has_ggv) {
    promoted_dtype = at::promote_types(promoted_dtype, ggv.scalar_type());
  }
  if (has_ggz) {
    promoted_dtype = at::promote_types(promoted_dtype, ggz.scalar_type());
  }

  // Expand and convert inputs
  auto grad_expanded = grad_output.expand(output_size).to(promoted_dtype).contiguous();
  auto v_expanded = v.expand(output_size).to(promoted_dtype).contiguous();
  auto z_expanded = z.expand(output_size).to(promoted_dtype).contiguous();

  at::Tensor ggv_expanded, ggz_expanded;
  if (has_ggv) {
    ggv_expanded = ggv.expand(output_size).to(promoted_dtype).contiguous();
  } else {
    ggv_expanded = at::zeros(output_size, grad_output.options().dtype(promoted_dtype));
  }
  if (has_ggz) {
    ggz_expanded = ggz.expand(output_size).to(promoted_dtype).contiguous();
  } else {
    ggz_expanded = at::zeros(output_size, grad_output.options().dtype(promoted_dtype));
  }

  // Allocate outputs
  auto grad_grad_output_out = at::empty(output_size, grad_output.options().dtype(promoted_dtype));
  auto grad_v_out = at::empty(output_size, grad_output.options().dtype(promoted_dtype));
  auto grad_z_out = at::empty(output_size, grad_output.options().dtype(promoted_dtype));

  int64_t numel = grad_expanded.numel();

  if (numel == 0) {
    return std::make_tuple(grad_grad_output_out, grad_v_out, grad_z_out);
  }

  const int threads = 256;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::kBFloat16, at::kHalf,
      promoted_dtype,
      "chebyshev_polynomial_t_backward_backward_cuda",
      [&]() {
        chebyshev_polynomial_t_backward_backward_kernel<scalar_t><<<blocks, threads, 0,
            at::cuda::getCurrentCUDAStream()>>>(
            ggv_expanded.data_ptr<scalar_t>(),
            ggz_expanded.data_ptr<scalar_t>(),
            grad_expanded.data_ptr<scalar_t>(),
            v_expanded.data_ptr<scalar_t>(),
            z_expanded.data_ptr<scalar_t>(),
            grad_grad_output_out.data_ptr<scalar_t>(),
            grad_v_out.data_ptr<scalar_t>(),
            grad_z_out.data_ptr<scalar_t>(),
            numel,
            has_ggv,
            has_ggz);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return std::make_tuple(grad_grad_output_out, grad_v_out, grad_z_out);
}

}  // namespace torchscience::cuda::special_functions
