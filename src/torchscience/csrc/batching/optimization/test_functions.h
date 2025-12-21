#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::batching::optimization::test_functions {

namespace {

/**
 * Move a batch dimension to the front of the tensor.
 * If batch_dim is nullopt, the tensor is returned unchanged.
 */
inline at::Tensor moveBatchDimToFront(const at::Tensor& tensor, std::optional<int64_t> batch_dim) {
    if (!batch_dim.has_value()) {
        return tensor;
    }
    int64_t bdim = batch_dim.value();
    if (bdim == 0) {
        return tensor;
    }
    return tensor.movedim(bdim, 0);
}

}  // namespace

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
    auto x_ = moveBatchDimToFront(x, x_bdim);
    auto a_ = moveBatchDimToFront(a, a_bdim);
    auto b_ = moveBatchDimToFront(b, b_bdim);

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
