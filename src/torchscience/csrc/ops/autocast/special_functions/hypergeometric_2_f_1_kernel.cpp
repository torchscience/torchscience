#include "../../special_functions.h"

#include <ATen/autocast_mode.h>
#include <torch/library.h>


namespace science::ops {
namespace {

at::Tensor
hypergeometric_2_f_1_autocast(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c,
    const at::Tensor& z
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    const at::Tensor a_promoted = at::autocast::cached_cast(at::kFloat, a);
    const at::Tensor b_promoted = at::autocast::cached_cast(at::kFloat, b);
    const at::Tensor c_promoted = at::autocast::cached_cast(at::kFloat, c);
    const at::Tensor z_promoted = at::autocast::cached_cast(at::kFloat, z);

    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
        "torchscience::hypergeometric_2_f_1",
        ""
    ).typed<at::Tensor(
        const at::Tensor&,
        const at::Tensor&,
        const at::Tensor&,
        const at::Tensor&
    )>();

    return op.call(
        a_promoted,
        b_promoted,
        c_promoted,
        z_promoted
    );
}

}  // namespace
} // namespace science::ops


// Register AutocastCUDA implementation
TORCH_LIBRARY_IMPL(torchscience, AutocastCUDA, module) {
    module.impl(
        TORCH_SELECTIVE_NAME(
            "torchscience::hypergeometric_2_f_1"
        ),
        TORCH_FN(
            science::ops::hypergeometric_2_f_1_autocast
        )
    );
}
