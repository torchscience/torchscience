#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::cpu {

inline at::Tensor incomplete_beta_forward(
  const at::Tensor &a,
  const at::Tensor &b,
  const at::Tensor &x
) {
  TORCH_CHECK(false, "incomplete_beta not yet implemented");
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> incomplete_beta_backward(
  const at::Tensor &grad,
  const at::Tensor &a,
  const at::Tensor &b,
  const at::Tensor &x
) {
  TORCH_CHECK(false, "incomplete_beta_backward not yet implemented");
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> incomplete_beta_backward_backward(
  const at::Tensor &gg_a,
  const at::Tensor &gg_b,
  const at::Tensor &gg_x,
  const at::Tensor &grad,
  const at::Tensor &a,
  const at::Tensor &b,
  const at::Tensor &x
) {
  TORCH_CHECK(false, "incomplete_beta_backward_backward not yet implemented");
}

} // namespace torchscience::cpu

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
  module.impl("incomplete_beta", torchscience::cpu::incomplete_beta_forward);

  module.impl("incomplete_beta_backward", torchscience::cpu::incomplete_beta_backward);

  module.impl("incomplete_beta_backward_backward", torchscience::cpu::incomplete_beta_backward_backward);
}
