#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "../../kernel/probability/gamma_cdf.h"
#include "../../kernel/probability/gamma_cdf_backward.h"
#include "../../kernel/probability/gamma_pdf.h"
#include "../../kernel/probability/gamma_pdf_backward.h"
#include "../../kernel/probability/gamma_ppf.h"
#include "../../kernel/probability/gamma_ppf_backward.h"

namespace torchscience::cpu::probability {

// reduce_grad is defined in normal.h and already available

at::Tensor gamma_cdf(
    const at::Tensor& x,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  auto tensors = at::broadcast_tensors({x, shape, scale});
  auto x_b = tensors[0].contiguous();
  auto shape_b = tensors[1].contiguous();
  auto scale_b = tensors[2].contiguous();

  auto output = at::empty_like(x_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "gamma_cdf_cpu", [&] {
        auto x_data = x_b.data_ptr<scalar_t>();
        auto shape_data = shape_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::gamma_cdf<scalar_t>(
                x_data[i], shape_data[i], scale_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> gamma_cdf_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  auto tensors = at::broadcast_tensors({grad, x, shape, scale});
  auto grad_b = tensors[0].contiguous();
  auto x_b = tensors[1].contiguous();
  auto shape_b = tensors[2].contiguous();
  auto scale_b = tensors[3].contiguous();

  auto grad_x = at::empty_like(x_b);
  auto grad_shape = at::empty_like(shape_b);
  auto grad_scale = at::empty_like(scale_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "gamma_cdf_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto x_data = x_b.data_ptr<scalar_t>();
        auto shape_data = shape_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto grad_x_data = grad_x.data_ptr<scalar_t>();
        auto grad_shape_data = grad_shape.data_ptr<scalar_t>();
        auto grad_scale_data = grad_scale.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gx, gs, gsc] = kernel::probability::gamma_cdf_backward<scalar_t>(
                grad_data[i], x_data[i], shape_data[i], scale_data[i]);
            grad_x_data[i] = gx;
            grad_shape_data[i] = gs;
            grad_scale_data[i] = gsc;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_x, x),
      reduce_grad(grad_shape, shape),
      reduce_grad(grad_scale, scale));
}

at::Tensor gamma_pdf(
    const at::Tensor& x,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  auto tensors = at::broadcast_tensors({x, shape, scale});
  auto x_b = tensors[0].contiguous();
  auto shape_b = tensors[1].contiguous();
  auto scale_b = tensors[2].contiguous();

  auto output = at::empty_like(x_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "gamma_pdf_cpu", [&] {
        auto x_data = x_b.data_ptr<scalar_t>();
        auto shape_data = shape_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::gamma_pdf<scalar_t>(
                x_data[i], shape_data[i], scale_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> gamma_pdf_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  auto tensors = at::broadcast_tensors({grad, x, shape, scale});
  auto grad_b = tensors[0].contiguous();
  auto x_b = tensors[1].contiguous();
  auto shape_b = tensors[2].contiguous();
  auto scale_b = tensors[3].contiguous();

  auto grad_x = at::empty_like(x_b);
  auto grad_shape = at::empty_like(shape_b);
  auto grad_scale = at::empty_like(scale_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "gamma_pdf_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto x_data = x_b.data_ptr<scalar_t>();
        auto shape_data = shape_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto grad_x_data = grad_x.data_ptr<scalar_t>();
        auto grad_shape_data = grad_shape.data_ptr<scalar_t>();
        auto grad_scale_data = grad_scale.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gx, gs, gsc] = kernel::probability::gamma_pdf_backward<scalar_t>(
                grad_data[i], x_data[i], shape_data[i], scale_data[i]);
            grad_x_data[i] = gx;
            grad_shape_data[i] = gs;
            grad_scale_data[i] = gsc;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_x, x),
      reduce_grad(grad_shape, shape),
      reduce_grad(grad_scale, scale));
}

at::Tensor gamma_ppf(
    const at::Tensor& p,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  auto tensors = at::broadcast_tensors({p, shape, scale});
  auto p_b = tensors[0].contiguous();
  auto shape_b = tensors[1].contiguous();
  auto scale_b = tensors[2].contiguous();

  auto output = at::empty_like(p_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, p.scalar_type(), "gamma_ppf_cpu", [&] {
        auto p_data = p_b.data_ptr<scalar_t>();
        auto shape_data = shape_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::gamma_ppf<scalar_t>(
                p_data[i], shape_data[i], scale_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> gamma_ppf_backward(
    const at::Tensor& grad,
    const at::Tensor& p,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  auto tensors = at::broadcast_tensors({grad, p, shape, scale});
  auto grad_b = tensors[0].contiguous();
  auto p_b = tensors[1].contiguous();
  auto shape_b = tensors[2].contiguous();
  auto scale_b = tensors[3].contiguous();

  auto grad_p = at::empty_like(p_b);
  auto grad_shape = at::empty_like(shape_b);
  auto grad_scale = at::empty_like(scale_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, p.scalar_type(), "gamma_ppf_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto p_data = p_b.data_ptr<scalar_t>();
        auto shape_data = shape_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto grad_p_data = grad_p.data_ptr<scalar_t>();
        auto grad_shape_data = grad_shape.data_ptr<scalar_t>();
        auto grad_scale_data = grad_scale.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gp, gs, gsc] = kernel::probability::gamma_ppf_backward<scalar_t>(
                grad_data[i], p_data[i], shape_data[i], scale_data[i]);
            grad_p_data[i] = gp;
            grad_shape_data[i] = gs;
            grad_scale_data[i] = gsc;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_p, p),
      reduce_grad(grad_shape, shape),
      reduce_grad(grad_scale, scale));
}

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("gamma_cdf", &gamma_cdf);
  m.impl("gamma_cdf_backward", &gamma_cdf_backward);
  m.impl("gamma_pdf", &gamma_pdf);
  m.impl("gamma_pdf_backward", &gamma_pdf_backward);
  m.impl("gamma_ppf", &gamma_ppf);
  m.impl("gamma_ppf_backward", &gamma_ppf_backward);
}

}  // namespace torchscience::cpu::probability
