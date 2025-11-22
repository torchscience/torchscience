#include <ATen/autocast_mode.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/DispatchKey.h>
#include <torch/library.h>
#include <torch/types.h>

namespace science {
namespace ops {
namespace {

// Autocast implementation for the example operator
at::Tensor example_autocast(const at::Tensor& input, const at::Scalar& x) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    // Call the actual example operator through the dispatcher
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("torchscience::example", "")
                         .typed<at::Tensor(const at::Tensor&, const at::Scalar&)>();

    // Cast input to float32, call operator, cast back
    return op.call(at::autocast::cached_cast(at::kFloat, input), x).to(input.scalar_type());
}

}  // namespace

TORCH_LIBRARY_IMPL(torchscience, Autocast, module) {
    module.impl(TORCH_SELECTIVE_NAME("torchscience::example"), TORCH_FN(example_autocast));
}

}  // namespace ops
}  // namespace science
