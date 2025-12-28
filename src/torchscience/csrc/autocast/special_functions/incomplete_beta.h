#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast {

inline at::Tensor incomplete_beta(
  const at::Tensor &a,
  const at::Tensor &b,
  const at::Tensor &x
) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(
    c10::DispatchKey::Autocast
  );

  auto dtype = at::autocast::promote_type(
    at::kFloat,
    a.device().type(),
    a,
    b,
    x
  );

  return c10::Dispatcher::singleton()
    .findSchemaOrThrow(
      "torchscience::incomplete_beta",
      ""
    ).typed<at::Tensor(
      const at::Tensor &,
      const at::Tensor &,
      const at::Tensor &
    )>()
    .call(
      at::autocast::cached_cast(dtype, a),
      at::autocast::cached_cast(dtype, b),
      at::autocast::cached_cast(dtype, x)
    );
}

} // namespace torchscience::autocast

TORCH_LIBRARY_IMPL(torchscience, Autocast, module) {
  module.impl(
    "incomplete_beta",
    torchscience::autocast::incomplete_beta
  );
}
