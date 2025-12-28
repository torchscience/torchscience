#pragma once

#include <torch/extension.h>

namespace torchscience::autograd {

inline at::Tensor incomplete_beta_autograd(const at::Tensor &a,
                                           const at::Tensor &b,
                                           const at::Tensor &x) {
  at::AutoDispatchBelowAutograd guard;
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchscience::incomplete_beta", "")
                       .typed<at::Tensor(const at::Tensor &, const at::Tensor &,
                                         const at::Tensor &)>();
  return op.call(a, b, x);
}

} // namespace torchscience::autograd

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
  m.impl("incomplete_beta", torchscience::autograd::incomplete_beta_autograd);
}
