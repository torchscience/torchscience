#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "../../kernel/probability/normal_cdf.h"
#include "../../kernel/probability/normal_cdf_backward.h"

namespace torchscience::cpu::probability {

// Helper to reduce gradient to match original input shape
inline at::Tensor reduce_grad(const at::Tensor& grad, const at::Tensor& input) {
  if (grad.sizes() == input.sizes()) {
    return grad;
  }
  // Sum along dimensions that were broadcast
  auto grad_reduced = grad;
  // Handle extra leading dimensions
  while (grad_reduced.dim() > input.dim()) {
    grad_reduced = grad_reduced.sum(0);
  }
  // Handle size-1 dimensions that were broadcast
  for (int64_t i = 0; i < input.dim(); ++i) {
    if (input.size(i) == 1 && grad_reduced.size(i) > 1) {
      grad_reduced = grad_reduced.sum(i, /*keepdim=*/true);
    }
  }
  return grad_reduced;
}

at::Tensor normal_cdf(
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  // Broadcast all inputs together
  auto tensors = at::broadcast_tensors({x, loc, scale});
  auto x_b = tensors[0].contiguous();
  auto loc_b = tensors[1].contiguous();
  auto scale_b = tensors[2].contiguous();

  auto output = at::empty_like(x_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "normal_cdf_cpu", [&] {
        auto x_data = x_b.data_ptr<scalar_t>();
        auto loc_data = loc_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::normal_cdf<scalar_t>(
                x_data[i], loc_data[i], scale_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> normal_cdf_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  // Broadcast all inputs
  auto tensors = at::broadcast_tensors({grad, x, loc, scale});
  auto grad_b = tensors[0].contiguous();
  auto x_b = tensors[1].contiguous();
  auto loc_b = tensors[2].contiguous();
  auto scale_b = tensors[3].contiguous();

  auto grad_x = at::empty_like(x_b);
  auto grad_loc = at::empty_like(loc_b);
  auto grad_scale = at::empty_like(scale_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "normal_cdf_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto x_data = x_b.data_ptr<scalar_t>();
        auto loc_data = loc_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto grad_x_data = grad_x.data_ptr<scalar_t>();
        auto grad_loc_data = grad_loc.data_ptr<scalar_t>();
        auto grad_scale_data = grad_scale.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            kernel::probability::normal_cdf_backward<scalar_t>(
                grad_data[i], x_data[i], loc_data[i], scale_data[i],
                grad_x_data[i], grad_loc_data[i], grad_scale_data[i]);
          }
        });
      });

  // Reduce gradients if inputs were broadcast
  return std::make_tuple(
      reduce_grad(grad_x, x),
      reduce_grad(grad_loc, loc),
      reduce_grad(grad_scale, scale));
}

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("normal_cdf", &normal_cdf);
  m.impl("normal_cdf_backward", &normal_cdf_backward);
}

}  // namespace torchscience::cpu::probability
