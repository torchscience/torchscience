#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast {

inline at::Tensor chebyshev_polynomial_t(const at::Tensor& x, const at::Tensor& n) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    static auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow(
            "torchscience::chebyshev_polynomial_t",
            ""
        )
        .typed<at::Tensor(
            const at::Tensor&,
            const at::Tensor&
        )>();

    auto dtype = at::autocast::promote_type(at::kFloat, x.device().type(), x, n);

    return op.call(
        at::autocast::cached_cast(dtype, x),
        at::autocast::cached_cast(dtype, n)
    );
}

}  // namespace torchscience::autocast

TORCH_LIBRARY_IMPL(torchscience, Autocast, module) {
    module.impl(
        "chebyshev_polynomial_t",
        torchscience::autocast::chebyshev_polynomial_t
    );
}
