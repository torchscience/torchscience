#pragma once

#include <vector>
#include <torch/extension.h>
#include "../../cpu/creation_operators.h"
#include "../../meta/creation_operators.h"

namespace torchscience::window_function {

namespace {

struct RectangularWindowTraits {
    static std::vector<int64_t> output_shape(int64_t n) {
        return {n};
    }

    template<typename scalar_t>
    static void kernel(scalar_t* output, int64_t numel, int64_t n) {
        (void)n;  // n == numel for rectangular window
        for (int64_t i = 0; i < numel; ++i) {
            output[i] = scalar_t(1);
        }
    }
};

}  // anonymous namespace

// Composite implementation that routes to appropriate backend based on device
inline at::Tensor rectangular_window(
  int64_t n,
  const c10::optional<at::ScalarType> dtype,
  const c10::optional<at::Layout> layout,
  const c10::optional<at::Device> device,
  const bool requires_grad
) {
  // Determine target device
  at::Device target_device = device.value_or(at::kCPU);

  // Dispatch to appropriate implementation
  if (target_device.type() == at::kMeta) {
    return torchscience::meta::MetaCreationOperator<RectangularWindowTraits>::forward<int64_t>(
      n, dtype, layout, device, requires_grad
    );
  } else {
    return torchscience::cpu::CPUCreationOperator<RectangularWindowTraits>::forward<int64_t>(
      n, dtype, layout, device, requires_grad
    );
  }
}

} // namespace torchscience::window_function

// CompositeExplicitAutograd implementation for operators with no tensor arguments
TORCH_LIBRARY_IMPL(torchscience, CompositeExplicitAutograd, module) {
  module.impl("rectangular_window", &torchscience::window_function::rectangular_window);
}
