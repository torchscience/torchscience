#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::quantized::cpu::integral_transform {

/**
 * Quantized CPU implementation of Hilbert transform.
 *
 * For quantized input, we dequantize before computation.
 * The output is a float tensor since FFT operations require floating-point.
 */
inline at::Tensor hilbert_transform(
    const at::Tensor& input,
    [[maybe_unused]] int64_t n_param,
    int64_t dim
) {
    TORCH_CHECK(
        input.is_quantized(),
        "hilbert_transform (QuantizedCPU) expects quantized tensor"
    );

    at::Tensor input_dequant = input.dequantize();

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::hilbert_transform", "")
        .typed<at::Tensor(const at::Tensor&, int64_t, int64_t)>()
        .call(input_dequant, n_param, dim);
}

/**
 * Backward pass for quantized CPU Hilbert transform.
 */
inline at::Tensor hilbert_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    [[maybe_unused]] int64_t n_param,
    int64_t dim
) {
    TORCH_CHECK(
        input.is_quantized(),
        "hilbert_transform_backward (QuantizedCPU) expects quantized tensor for input"
    );

    at::Tensor input_dequant = input.dequantize();

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::hilbert_transform_backward", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t, int64_t)>()
        .call(grad_output, input_dequant, n_param, dim);
}

/**
 * Double backward pass for quantized CPU Hilbert transform.
 */
inline std::tuple<at::Tensor, at::Tensor> hilbert_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    [[maybe_unused]] int64_t n_param,
    int64_t dim
) {
    at::Tensor input_dequant = input.is_quantized() ? input.dequantize() : input;

    auto [gg_output, new_grad_input] = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::hilbert_transform_backward_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t
        )>()
        .call(grad_grad_input, grad_output, input_dequant, n_param, dim);

    return {gg_output, new_grad_input};
}

}  // namespace torchscience::quantized::cpu::integral_transform

TORCH_LIBRARY_IMPL(torchscience, QuantizedCPU, module) {
    module.impl(
        "hilbert_transform",
        &torchscience::quantized::cpu::integral_transform::hilbert_transform
    );
    module.impl(
        "hilbert_transform_backward",
        &torchscience::quantized::cpu::integral_transform::hilbert_transform_backward
    );
    module.impl(
        "hilbert_transform_backward_backward",
        &torchscience::quantized::cpu::integral_transform::hilbert_transform_backward_backward
    );
}
