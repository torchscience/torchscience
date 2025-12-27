#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast {

inline at::Tensor hypergeometric_2_f_1_autocast(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c,
    const at::Tensor& z
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
    static auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::hypergeometric_2_f_1", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>();
    auto dtype = at::autocast::promote_type(at::kFloat, a, b, c, z);
    return op.call(
        at::autocast::cached_cast(dtype, a),
        at::autocast::cached_cast(dtype, b),
        at::autocast::cached_cast(dtype, c),
        at::autocast::cached_cast(dtype, z));
}

}  // namespace torchscience::autocast

TORCH_LIBRARY_IMPL(torchscience, Autocast, m) {
    m.impl("hypergeometric_2_f_1", torchscience::autocast::hypergeometric_2_f_1_autocast);
}
