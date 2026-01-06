#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "../../kernel/probability/f_cdf.h"
#include "../../kernel/probability/f_cdf_backward.h"
#include "../../kernel/probability/f_pdf.h"
#include "../../kernel/probability/f_pdf_backward.h"
#include "../../kernel/probability/f_ppf.h"
#include "../../kernel/probability/f_ppf_backward.h"
#include "../../kernel/probability/f_sf.h"
#include "../../kernel/probability/f_sf_backward.h"

namespace torchscience::cpu::probability {

// reduce_grad is defined in normal.h and already available in this namespace

at::Tensor f_cdf(
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  auto tensors = at::broadcast_tensors({x, dfn, dfd});
  auto x_b = tensors[0].contiguous();
  auto dfn_b = tensors[1].contiguous();
  auto dfd_b = tensors[2].contiguous();

  auto output = at::empty_like(x_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "f_cdf_cpu", [&] {
        auto x_data = x_b.data_ptr<scalar_t>();
        auto dfn_data = dfn_b.data_ptr<scalar_t>();
        auto dfd_data = dfd_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::f_cdf<scalar_t>(
                x_data[i], dfn_data[i], dfd_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> f_cdf_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  auto tensors = at::broadcast_tensors({grad, x, dfn, dfd});
  auto grad_b = tensors[0].contiguous();
  auto x_b = tensors[1].contiguous();
  auto dfn_b = tensors[2].contiguous();
  auto dfd_b = tensors[3].contiguous();

  auto grad_x = at::empty_like(x_b);
  auto grad_dfn = at::empty_like(dfn_b);
  auto grad_dfd = at::empty_like(dfd_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "f_cdf_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto x_data = x_b.data_ptr<scalar_t>();
        auto dfn_data = dfn_b.data_ptr<scalar_t>();
        auto dfd_data = dfd_b.data_ptr<scalar_t>();
        auto grad_x_data = grad_x.data_ptr<scalar_t>();
        auto grad_dfn_data = grad_dfn.data_ptr<scalar_t>();
        auto grad_dfd_data = grad_dfd.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gx, gdfn, gdfd] = kernel::probability::f_cdf_backward<scalar_t>(
                grad_data[i], x_data[i], dfn_data[i], dfd_data[i]);
            grad_x_data[i] = gx;
            grad_dfn_data[i] = gdfn;
            grad_dfd_data[i] = gdfd;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_x, x),
      reduce_grad(grad_dfn, dfn),
      reduce_grad(grad_dfd, dfd));
}

at::Tensor f_pdf(
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  auto tensors = at::broadcast_tensors({x, dfn, dfd});
  auto x_b = tensors[0].contiguous();
  auto dfn_b = tensors[1].contiguous();
  auto dfd_b = tensors[2].contiguous();

  auto output = at::empty_like(x_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "f_pdf_cpu", [&] {
        auto x_data = x_b.data_ptr<scalar_t>();
        auto dfn_data = dfn_b.data_ptr<scalar_t>();
        auto dfd_data = dfd_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::f_pdf<scalar_t>(
                x_data[i], dfn_data[i], dfd_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> f_pdf_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  auto tensors = at::broadcast_tensors({grad, x, dfn, dfd});
  auto grad_b = tensors[0].contiguous();
  auto x_b = tensors[1].contiguous();
  auto dfn_b = tensors[2].contiguous();
  auto dfd_b = tensors[3].contiguous();

  auto grad_x = at::empty_like(x_b);
  auto grad_dfn = at::empty_like(dfn_b);
  auto grad_dfd = at::empty_like(dfd_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "f_pdf_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto x_data = x_b.data_ptr<scalar_t>();
        auto dfn_data = dfn_b.data_ptr<scalar_t>();
        auto dfd_data = dfd_b.data_ptr<scalar_t>();
        auto grad_x_data = grad_x.data_ptr<scalar_t>();
        auto grad_dfn_data = grad_dfn.data_ptr<scalar_t>();
        auto grad_dfd_data = grad_dfd.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gx, gdfn, gdfd] = kernel::probability::f_pdf_backward<scalar_t>(
                grad_data[i], x_data[i], dfn_data[i], dfd_data[i]);
            grad_x_data[i] = gx;
            grad_dfn_data[i] = gdfn;
            grad_dfd_data[i] = gdfd;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_x, x),
      reduce_grad(grad_dfn, dfn),
      reduce_grad(grad_dfd, dfd));
}

at::Tensor f_ppf(
    const at::Tensor& p,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  auto tensors = at::broadcast_tensors({p, dfn, dfd});
  auto p_b = tensors[0].contiguous();
  auto dfn_b = tensors[1].contiguous();
  auto dfd_b = tensors[2].contiguous();

  auto output = at::empty_like(p_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, p.scalar_type(), "f_ppf_cpu", [&] {
        auto p_data = p_b.data_ptr<scalar_t>();
        auto dfn_data = dfn_b.data_ptr<scalar_t>();
        auto dfd_data = dfd_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::f_ppf<scalar_t>(
                p_data[i], dfn_data[i], dfd_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> f_ppf_backward(
    const at::Tensor& grad,
    const at::Tensor& p,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  auto tensors = at::broadcast_tensors({grad, p, dfn, dfd});
  auto grad_b = tensors[0].contiguous();
  auto p_b = tensors[1].contiguous();
  auto dfn_b = tensors[2].contiguous();
  auto dfd_b = tensors[3].contiguous();

  auto grad_p = at::empty_like(p_b);
  auto grad_dfn = at::empty_like(dfn_b);
  auto grad_dfd = at::empty_like(dfd_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, p.scalar_type(), "f_ppf_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto p_data = p_b.data_ptr<scalar_t>();
        auto dfn_data = dfn_b.data_ptr<scalar_t>();
        auto dfd_data = dfd_b.data_ptr<scalar_t>();
        auto grad_p_data = grad_p.data_ptr<scalar_t>();
        auto grad_dfn_data = grad_dfn.data_ptr<scalar_t>();
        auto grad_dfd_data = grad_dfd.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gp, gdfn, gdfd] = kernel::probability::f_ppf_backward<scalar_t>(
                grad_data[i], p_data[i], dfn_data[i], dfd_data[i]);
            grad_p_data[i] = gp;
            grad_dfn_data[i] = gdfn;
            grad_dfd_data[i] = gdfd;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_p, p),
      reduce_grad(grad_dfn, dfn),
      reduce_grad(grad_dfd, dfd));
}

at::Tensor f_sf(
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  auto tensors = at::broadcast_tensors({x, dfn, dfd});
  auto x_b = tensors[0].contiguous();
  auto dfn_b = tensors[1].contiguous();
  auto dfd_b = tensors[2].contiguous();

  auto output = at::empty_like(x_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "f_sf_cpu", [&] {
        auto x_data = x_b.data_ptr<scalar_t>();
        auto dfn_data = dfn_b.data_ptr<scalar_t>();
        auto dfd_data = dfd_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::f_sf<scalar_t>(
                x_data[i], dfn_data[i], dfd_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> f_sf_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  auto tensors = at::broadcast_tensors({grad, x, dfn, dfd});
  auto grad_b = tensors[0].contiguous();
  auto x_b = tensors[1].contiguous();
  auto dfn_b = tensors[2].contiguous();
  auto dfd_b = tensors[3].contiguous();

  auto grad_x = at::empty_like(x_b);
  auto grad_dfn = at::empty_like(dfn_b);
  auto grad_dfd = at::empty_like(dfd_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "f_sf_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto x_data = x_b.data_ptr<scalar_t>();
        auto dfn_data = dfn_b.data_ptr<scalar_t>();
        auto dfd_data = dfd_b.data_ptr<scalar_t>();
        auto grad_x_data = grad_x.data_ptr<scalar_t>();
        auto grad_dfn_data = grad_dfn.data_ptr<scalar_t>();
        auto grad_dfd_data = grad_dfd.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gx, gdfn, gdfd] = kernel::probability::f_sf_backward<scalar_t>(
                grad_data[i], x_data[i], dfn_data[i], dfd_data[i]);
            grad_x_data[i] = gx;
            grad_dfn_data[i] = gdfn;
            grad_dfd_data[i] = gdfd;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_x, x),
      reduce_grad(grad_dfn, dfn),
      reduce_grad(grad_dfd, dfd));
}

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("f_cdf", &f_cdf);
  m.impl("f_cdf_backward", &f_cdf_backward);
  m.impl("f_pdf", &f_pdf);
  m.impl("f_pdf_backward", &f_pdf_backward);
  m.impl("f_ppf", &f_ppf);
  m.impl("f_ppf_backward", &f_ppf_backward);
  m.impl("f_sf", &f_sf);
  m.impl("f_sf_backward", &f_sf_backward);
}

}  // namespace torchscience::cpu::probability
