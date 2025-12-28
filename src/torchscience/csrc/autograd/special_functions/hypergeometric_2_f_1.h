#pragma once

#include <torch/extension.h>

namespace torchscience::autograd {

inline at::Tensor hypergeometric_2_f_1_autograd(const at::Tensor &a,
                                                const at::Tensor &b,
                                                const at::Tensor &c,
                                                const at::Tensor &z) {
  at::AutoDispatchBelowAutograd guard;
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchscience::hypergeometric_2_f_1", "")
          .typed<at::Tensor(const at::Tensor &, const at::Tensor &,
                            const at::Tensor &, const at::Tensor &)>();
  return op.call(a, b, c, z);
}

} // namespace torchscience::autograd

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
  m.impl("hypergeometric_2_f_1",
         torchscience::autograd::hypergeometric_2_f_1_autograd);
}
