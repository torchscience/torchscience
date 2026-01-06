#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "../../kernel/probability/chi2_cdf.h"
#include "../../kernel/probability/chi2_cdf_backward.h"

namespace torchscience::cpu::probability {

// reduce_grad is defined in normal.h and already available in this namespace

at::Tensor chi2_cdf(
    const at::Tensor& x,
    const at::Tensor& df) {
  // Broadcast inputs together
  auto tensors = at::broadcast_tensors({x, df});
  auto x_b = tensors[0].contiguous();
  auto df_b = tensors[1].contiguous();

  auto output = at::empty_like(x_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "chi2_cdf_cpu", [&] {
        auto x_data = x_b.data_ptr<scalar_t>();
        auto df_data = df_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::chi2_cdf<scalar_t>(
                x_data[i], df_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor> chi2_cdf_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& df) {
  // Broadcast all inputs
  auto tensors = at::broadcast_tensors({grad, x, df});
  auto grad_b = tensors[0].contiguous();
  auto x_b = tensors[1].contiguous();
  auto df_b = tensors[2].contiguous();

  auto grad_x = at::empty_like(x_b);
  auto grad_df = at::empty_like(df_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "chi2_cdf_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto x_data = x_b.data_ptr<scalar_t>();
        auto df_data = df_b.data_ptr<scalar_t>();
        auto grad_x_data = grad_x.data_ptr<scalar_t>();
        auto grad_df_data = grad_df.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gx, gdf] = kernel::probability::chi2_cdf_backward<scalar_t>(
                grad_data[i], x_data[i], df_data[i]);
            grad_x_data[i] = gx;
            grad_df_data[i] = gdf;
          }
        });
      });

  // Reduce gradients if inputs were broadcast
  return std::make_tuple(
      reduce_grad(grad_x, x),
      reduce_grad(grad_df, df));
}

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("chi2_cdf", &chi2_cdf);
  m.impl("chi2_cdf_backward", &chi2_cdf_backward);
}

}  // namespace torchscience::cpu::probability
