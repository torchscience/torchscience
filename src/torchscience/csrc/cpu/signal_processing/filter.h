#pragma once

#include <tuple>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <torch/library.h>

#include "../../impl/signal_processing/filter/butterworth_analog_bandpass_filter.h"
#include "../../impl/signal_processing/filter/butterworth_analog_bandpass_filter_backward.h"
#include "../../impl/signal_processing/filter/butterworth_analog_bandpass_filter_backward_backward.h"

namespace torchscience::cpu::filter {

/**
 * CPU implementation of butterworth_analog_bandpass_filter.
 *
 * This is NOT an element-wise operation: output shape is (*batch_shape, n, 6)
 * where n is the filter order.
 *
 * @param n Filter order (positive integer)
 * @param omega_p1 Lower passband frequency tensor, shape (*batch_shape)
 * @param omega_p2 Upper passband frequency tensor, shape (*batch_shape)
 * @return SOS coefficients tensor, shape (*batch_shape, n, 6)
 */
inline at::Tensor butterworth_analog_bandpass_filter(
    int64_t n,
    const at::Tensor& omega_p1,
    const at::Tensor& omega_p2
) {
    TORCH_CHECK(n > 0, "butterworth_analog_bandpass_filter: order n must be positive, got ", n);
    TORCH_CHECK(n <= 64, "butterworth_analog_bandpass_filter: order n must be <= 64, got ", n);

    // Broadcast omega_p1 and omega_p2 to common shape
    auto broadcasted = at::broadcast_tensors({omega_p1, omega_p2});
    at::Tensor omega_p1_bc = broadcasted[0].contiguous();
    at::Tensor omega_p2_bc = broadcasted[1].contiguous();

    // Get batch shape
    auto batch_shape = omega_p1_bc.sizes().vec();

    // Compute output shape: (*batch_shape, n, 6)
    std::vector<int64_t> output_shape = batch_shape;
    output_shape.push_back(n);
    output_shape.push_back(6);

    // Create output tensor
    auto options = omega_p1_bc.options();
    at::Tensor output = at::empty(output_shape, options);

    // Flatten batch dimensions for parallel processing
    int64_t batch_size = omega_p1_bc.numel();
    at::Tensor omega_p1_flat = omega_p1_bc.flatten();
    at::Tensor omega_p2_flat = omega_p2_bc.flatten();
    at::Tensor output_flat = output.view({batch_size, n, 6});

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        omega_p1_flat.scalar_type(),
        "butterworth_analog_bandpass_filter_cpu",
        [&]() {
            const scalar_t* omega_p1_data = omega_p1_flat.data_ptr<scalar_t>();
            const scalar_t* omega_p2_data = omega_p2_flat.data_ptr<scalar_t>();
            scalar_t* output_data = output_flat.data_ptr<scalar_t>();

            // Parallel over batch dimension
            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    scalar_t w1 = omega_p1_data[i];
                    scalar_t w2 = omega_p2_data[i];

                    // Validate frequencies
                    // Allow any positive frequencies where w1 < w2
                    // (normalized to Nyquist is handled at Python level)

                    // Compute SOS for this batch element
                    scalar_t* sos_ptr = output_data + i * n * 6;

                    // Use float for computation if half precision
                    if constexpr (std::is_same_v<scalar_t, at::Half> ||
                                  std::is_same_v<scalar_t, at::BFloat16>) {
                        float w1_f = static_cast<float>(w1);
                        float w2_f = static_cast<float>(w2);
                        float sos_f[64 * 6];  // Max order * 6

                        impl::filter::butterworth_analog_bandpass_filter<float>(
                            n, w1_f, w2_f, sos_f
                        );

                        // Convert back to scalar_t
                        for (int64_t j = 0; j < n * 6; ++j) {
                            sos_ptr[j] = static_cast<scalar_t>(sos_f[j]);
                        }
                    } else {
                        impl::filter::butterworth_analog_bandpass_filter<scalar_t>(
                            n, w1, w2, sos_ptr
                        );
                    }
                }
            });
        }
    );

    return output;
}

/**
 * Backward pass for butterworth_analog_bandpass_filter on CPU.
 */
inline std::tuple<at::Tensor, at::Tensor> butterworth_analog_bandpass_filter_backward(
    const at::Tensor& grad_output,
    int64_t n,
    const at::Tensor& omega_p1,
    const at::Tensor& omega_p2
) {
    // Broadcast omega_p1 and omega_p2 to common shape
    auto broadcasted = at::broadcast_tensors({omega_p1, omega_p2});
    at::Tensor omega_p1_bc = broadcasted[0].contiguous();
    at::Tensor omega_p2_bc = broadcasted[1].contiguous();

    // Get batch shape
    auto batch_shape = omega_p1_bc.sizes().vec();
    int64_t batch_size = omega_p1_bc.numel();

    // Flatten for processing
    at::Tensor omega_p1_flat = omega_p1_bc.flatten();
    at::Tensor omega_p2_flat = omega_p2_bc.flatten();

    // grad_output shape: (*batch_shape, n, 6)
    at::Tensor grad_output_flat = grad_output.view({batch_size, n, 6}).contiguous();

    // Create output gradients
    at::Tensor grad_omega_p1_flat = at::empty({batch_size}, omega_p1_flat.options());
    at::Tensor grad_omega_p2_flat = at::empty({batch_size}, omega_p2_flat.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        omega_p1_flat.scalar_type(),
        "butterworth_analog_bandpass_filter_backward_cpu",
        [&]() {
            const scalar_t* omega_p1_data = omega_p1_flat.data_ptr<scalar_t>();
            const scalar_t* omega_p2_data = omega_p2_flat.data_ptr<scalar_t>();
            const scalar_t* grad_output_data = grad_output_flat.data_ptr<scalar_t>();
            scalar_t* grad_omega_p1_data = grad_omega_p1_flat.data_ptr<scalar_t>();
            scalar_t* grad_omega_p2_data = grad_omega_p2_flat.data_ptr<scalar_t>();

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    scalar_t w1 = omega_p1_data[i];
                    scalar_t w2 = omega_p2_data[i];
                    const scalar_t* grad_sos = grad_output_data + i * n * 6;

                    scalar_t grad_w1, grad_w2;

                    if constexpr (std::is_same_v<scalar_t, at::Half> ||
                                  std::is_same_v<scalar_t, at::BFloat16>) {
                        float w1_f = static_cast<float>(w1);
                        float w2_f = static_cast<float>(w2);
                        float grad_sos_f[64 * 6];
                        for (int64_t j = 0; j < n * 6; ++j) {
                            grad_sos_f[j] = static_cast<float>(grad_sos[j]);
                        }
                        float grad_w1_f, grad_w2_f;

                        impl::filter::butterworth_analog_bandpass_filter_backward<float>(
                            grad_sos_f, n, w1_f, w2_f, grad_w1_f, grad_w2_f
                        );

                        grad_w1 = static_cast<scalar_t>(grad_w1_f);
                        grad_w2 = static_cast<scalar_t>(grad_w2_f);
                    } else {
                        impl::filter::butterworth_analog_bandpass_filter_backward<scalar_t>(
                            grad_sos, n, w1, w2, grad_w1, grad_w2
                        );
                    }

                    grad_omega_p1_data[i] = grad_w1;
                    grad_omega_p2_data[i] = grad_w2;
                }
            });
        }
    );

    // Reshape back to original batch shape
    at::Tensor grad_omega_p1 = grad_omega_p1_flat.view(batch_shape);
    at::Tensor grad_omega_p2 = grad_omega_p2_flat.view(batch_shape);

    return std::make_tuple(grad_omega_p1, grad_omega_p2);
}

/**
 * Double-backward pass for butterworth_analog_bandpass_filter on CPU.
 *
 * Computes second-order gradients for Hessian-vector products.
 *
 * @param grad_grad_omega_p1 Gradient w.r.t. grad_omega_p1 from first backward
 * @param grad_grad_omega_p2 Gradient w.r.t. grad_omega_p2 from first backward
 * @param grad_output Original gradient w.r.t. SOS output
 * @param n Filter order
 * @param omega_p1 Lower passband frequency
 * @param omega_p2 Upper passband frequency
 * @return Tuple of (grad_grad_output, new_grad_omega_p1, new_grad_omega_p2)
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> butterworth_analog_bandpass_filter_backward_backward(
    const at::Tensor& grad_grad_omega_p1,
    const at::Tensor& grad_grad_omega_p2,
    const at::Tensor& grad_output,
    int64_t n,
    const at::Tensor& omega_p1,
    const at::Tensor& omega_p2
) {
    // Broadcast omega_p1 and omega_p2 to common shape
    auto broadcasted = at::broadcast_tensors({omega_p1, omega_p2});
    at::Tensor omega_p1_bc = broadcasted[0].contiguous();
    at::Tensor omega_p2_bc = broadcasted[1].contiguous();

    // Get batch shape
    auto batch_shape = omega_p1_bc.sizes().vec();
    int64_t batch_size = omega_p1_bc.numel();

    // Flatten for processing
    at::Tensor omega_p1_flat = omega_p1_bc.flatten();
    at::Tensor omega_p2_flat = omega_p2_bc.flatten();

    // grad_output shape: (*batch_shape, n, 6)
    at::Tensor grad_output_flat = grad_output.view({batch_size, n, 6}).contiguous();

    // Handle optional grad_grad inputs
    bool has_gg_omega_p1 = grad_grad_omega_p1.defined() && grad_grad_omega_p1.numel() > 0;
    bool has_gg_omega_p2 = grad_grad_omega_p2.defined() && grad_grad_omega_p2.numel() > 0;

    at::Tensor gg_omega_p1_flat, gg_omega_p2_flat;
    if (has_gg_omega_p1) {
        gg_omega_p1_flat = grad_grad_omega_p1.flatten().contiguous();
    }
    if (has_gg_omega_p2) {
        gg_omega_p2_flat = grad_grad_omega_p2.flatten().contiguous();
    }

    // Create outputs
    at::Tensor grad_grad_output_flat = at::zeros({batch_size, n, 6}, grad_output.options());
    at::Tensor new_grad_omega_p1_flat = at::zeros({batch_size}, omega_p1_flat.options());
    at::Tensor new_grad_omega_p2_flat = at::zeros({batch_size}, omega_p2_flat.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        omega_p1_flat.scalar_type(),
        "butterworth_analog_bandpass_filter_backward_backward_cpu",
        [&]() {
            const scalar_t* omega_p1_data = omega_p1_flat.data_ptr<scalar_t>();
            const scalar_t* omega_p2_data = omega_p2_flat.data_ptr<scalar_t>();
            const scalar_t* grad_output_data = grad_output_flat.data_ptr<scalar_t>();

            const scalar_t* gg_omega_p1_data = has_gg_omega_p1 ? gg_omega_p1_flat.data_ptr<scalar_t>() : nullptr;
            const scalar_t* gg_omega_p2_data = has_gg_omega_p2 ? gg_omega_p2_flat.data_ptr<scalar_t>() : nullptr;

            scalar_t* grad_grad_output_data = grad_grad_output_flat.data_ptr<scalar_t>();
            scalar_t* new_grad_omega_p1_data = new_grad_omega_p1_flat.data_ptr<scalar_t>();
            scalar_t* new_grad_omega_p2_data = new_grad_omega_p2_flat.data_ptr<scalar_t>();

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    scalar_t w1 = omega_p1_data[i];
                    scalar_t w2 = omega_p2_data[i];
                    const scalar_t* grad_sos = grad_output_data + i * n * 6;

                    scalar_t gg_w1 = has_gg_omega_p1 ? gg_omega_p1_data[i] : scalar_t(0);
                    scalar_t gg_w2 = has_gg_omega_p2 ? gg_omega_p2_data[i] : scalar_t(0);

                    scalar_t* grad_grad_sos = grad_grad_output_data + i * n * 6;
                    scalar_t new_grad_w1, new_grad_w2;

                    if constexpr (std::is_same_v<scalar_t, at::Half> ||
                                  std::is_same_v<scalar_t, at::BFloat16>) {
                        float w1_f = static_cast<float>(w1);
                        float w2_f = static_cast<float>(w2);
                        float gg_w1_f = static_cast<float>(gg_w1);
                        float gg_w2_f = static_cast<float>(gg_w2);

                        float grad_sos_f[64 * 6];
                        float grad_grad_sos_f[64 * 6];
                        for (int64_t j = 0; j < n * 6; ++j) {
                            grad_sos_f[j] = static_cast<float>(grad_sos[j]);
                        }

                        float new_grad_w1_f, new_grad_w2_f;

                        impl::filter::butterworth_analog_bandpass_filter_backward_backward<float>(
                            gg_w1_f,
                            gg_w2_f,
                            grad_sos_f,
                            n,
                            w1_f,
                            w2_f,
                            has_gg_omega_p1,
                            has_gg_omega_p2,
                            grad_grad_sos_f,
                            new_grad_w1_f,
                            new_grad_w2_f
                        );

                        for (int64_t j = 0; j < n * 6; ++j) {
                            grad_grad_sos[j] = static_cast<scalar_t>(grad_grad_sos_f[j]);
                        }
                        new_grad_w1 = static_cast<scalar_t>(new_grad_w1_f);
                        new_grad_w2 = static_cast<scalar_t>(new_grad_w2_f);
                    } else {
                        impl::filter::butterworth_analog_bandpass_filter_backward_backward<scalar_t>(
                            gg_w1,
                            gg_w2,
                            grad_sos,
                            n,
                            w1,
                            w2,
                            has_gg_omega_p1,
                            has_gg_omega_p2,
                            grad_grad_sos,
                            new_grad_w1,
                            new_grad_w2
                        );
                    }

                    new_grad_omega_p1_data[i] = new_grad_w1;
                    new_grad_omega_p2_data[i] = new_grad_w2;
                }
            });
        }
    );

    // Reshape back to original shapes
    at::Tensor grad_grad_output_reshaped = grad_grad_output_flat.view(grad_output.sizes());
    at::Tensor new_grad_omega_p1 = new_grad_omega_p1_flat.view(batch_shape);
    at::Tensor new_grad_omega_p2 = new_grad_omega_p2_flat.view(batch_shape);

    return std::make_tuple(grad_grad_output_reshaped, new_grad_omega_p1, new_grad_omega_p2);
}

}  // namespace torchscience::cpu::filter

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "butterworth_analog_bandpass_filter",
        &torchscience::cpu::filter::butterworth_analog_bandpass_filter
    );

    module.impl(
        "butterworth_analog_bandpass_filter_backward",
        &torchscience::cpu::filter::butterworth_analog_bandpass_filter_backward
    );

    module.impl(
        "butterworth_analog_bandpass_filter_backward_backward",
        &torchscience::cpu::filter::butterworth_analog_bandpass_filter_backward_backward
    );
}
