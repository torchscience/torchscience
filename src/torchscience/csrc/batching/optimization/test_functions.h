#pragma once

#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/PlumbingHelper.h>

namespace torchscience::batching::optimization::test_functions {

/**
 * Batching rule for the Rosenbrock function.
 *
 * The Rosenbrock function naturally supports batching - it operates on the
 * last dimension of the input tensor as coordinates, treating all leading
 * dimensions as batch dimensions. This batching rule handles the case where
 * vmap adds an additional batch dimension.
 *
 * For tensors x, a, b with optional batch dimensions at position bdim:
 * - If x has a batch dim, we move it to the front and let the function handle it
 * - Parameters a and b are broadcast appropriately
 */
inline std::tuple<at::Tensor, std::optional<int64_t>> rosenbrock_batch_rule(
    const at::Tensor& x,
    std::optional<int64_t> x_bdim,
    const at::Tensor& a,
    std::optional<int64_t> a_bdim,
    const at::Tensor& b,
    std::optional<int64_t> b_bdim
) {
    // Move batch dimensions to the front for uniform handling
    auto x_ = at::functorch::moveBatchDimToFront(x, x_bdim);
    auto a_ = at::functorch::moveBatchDimToFront(a, a_bdim);
    auto b_ = at::functorch::moveBatchDimToFront(b, b_bdim);

    // Call the underlying implementation through the dispatcher
    // The function already handles batch dimensions naturally
    at::AutoDispatchBelowAutograd guard;
    auto result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::rosenbrock", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(x_, a_, b_);

    // Result has batch dimension at front (position 0)
    return std::make_tuple(result, 0);
}

}  // namespace torchscience::batching::optimization::test_functions

TORCH_LIBRARY_IMPL(torchscience, FuncTorchBatched, m) {
    m.impl("rosenbrock", &torchscience::batching::optimization::test_functions::rosenbrock_batch_rule);
}
