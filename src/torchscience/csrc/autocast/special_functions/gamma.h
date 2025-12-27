#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast {

inline at::Tensor gamma(const at::Tensor& input) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    static auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow(
            "torchscience::gamma",
            ""
        )
        .typed<at::Tensor(
            const at::Tensor&
        )>();

    return op.call(
        at::autocast::cached_cast(at::kFloat, input)
    );
}

}  // namespace torchscience::autocast

TORCH_LIBRARY_IMPL(torchscience, Autocast, module) {
    module.impl(
        "gamma",
        torchscience::autocast::gamma
    );
}
