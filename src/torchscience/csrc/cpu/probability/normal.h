#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "../../kernel/probability/normal_cdf.h"

namespace torchscience::cpu::probability {

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

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("normal_cdf", &normal_cdf);
}

}  // namespace torchscience::cpu::probability
