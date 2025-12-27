#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast {

inline at::Tensor incomplete_beta_autocast(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& x
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
    static auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::incomplete_beta", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>();
    auto dtype = at::autocast::promote_type(at::kFloat, a, b, x);
    return op.call(
        at::autocast::cached_cast(dtype, a),
        at::autocast::cached_cast(dtype, b),
        at::autocast::cached_cast(dtype, x));
}

}  // namespace torchscience::autocast

TORCH_LIBRARY_IMPL(torchscience, Autocast, m) {
    m.impl("incomplete_beta", torchscience::autocast::incomplete_beta_autocast);
}
