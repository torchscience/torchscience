#include "../example.h"

#include <ATen/autocast_mode.h>
#include <torch/library.h>
#include <torch/types.h>

namespace science {
namespace ops {
namespace {
at::Tensor
example_autocast(
    const at::Tensor& input,
    int64_t foo,
    int64_t bar,
    int64_t baz,
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    return example(
        at::autocast::cached_cast(at::kFloat, input),
        foo,
        bar,
        baz,
    ).to(input.scalar_type()
  );
}
}

TORCH_LIBRARY_IMPL(torchscience, Autocast, module) {
  module.impl(
      TORCH_SELECTIVE_NAME("torchscience::example"),
      TORCH_FN(example_autocast)
  );
}
} // namespace ops
} // namespace science
