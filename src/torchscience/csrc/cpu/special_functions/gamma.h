#pragma once

#include <cmath>
#include <tuple>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/library.h>

namespace torchscience::cpu {

namespace {

template<typename T>
T gamma_kernel(T z) {
    constexpr T g = T(7);
    constexpr T coefficients[] = {
        T(0.99999999999980993),
        T(676.5203681218851),
        T(-1259.1392167224028),
        T(771.32342877765313),
        T(-176.61502916214059),
        T(12.507343278686905),
        T(-0.13857109526572012),
        T(9.9843695780195716e-6),
        T(1.5056327351493116e-7)
    };

    if (z < T(0.5)) {
        return T(M_PI) / (std::sin(T(M_PI) * z) * gamma_kernel(T(1) - z));
    }

    z -= T(1);
    T x = coefficients[0];
    for (int i = 1; i < 9; ++i) {
        x += coefficients[i] / (z + T(i));
    }

    T t = z + g + T(0.5);
    return std::sqrt(T(2 * M_PI)) * std::pow(t, z + T(0.5)) * std::exp(-t) * x;
}

template<typename T>
T digamma_kernel(T x) {
    T result = T(0);
    while (x < T(6)) {
        result -= T(1) / x;
        x += T(1);
    }
    T x2 = T(1) / (x * x);
    result += std::log(x) - T(0.5) / x
        - x2 * (T(1.0/12) - x2 * (T(1.0/120) - x2 * T(1.0/252)));
    return result;
}

template<typename T>
T trigamma_kernel(T x) {
    T result = T(0);
    while (x < T(6)) {
        result += T(1) / (x * x);
        x += T(1);
    }
    T x2 = T(1) / (x * x);
    result += T(1) / x + T(0.5) * x2
        + x2 / x * (T(1.0/6) - x2 * (T(1.0/30) - x2 * T(1.0/42)));
    return result;
}

}  // anonymous namespace

inline at::Tensor gamma_forward(const at::Tensor& input) {
    at::Tensor output;
    auto iter = at::TensorIteratorConfig()
        .add_output(output)
        .add_const_input(input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf, iter.common_dtype(), "gamma_cpu", [&] {
            at::native::cpu_kernel(iter, [](scalar_t x) -> scalar_t {
                return gamma_kernel(x);
            });
        });

    return iter.output();
}

inline at::Tensor gamma_backward(const at::Tensor& grad, const at::Tensor& input) {
    at::Tensor grad_input;
    auto iter = at::TensorIteratorConfig()
        .add_output(grad_input)
        .add_const_input(grad)
        .add_const_input(input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf, iter.common_dtype(), "gamma_backward_cpu", [&] {
            at::native::cpu_kernel(iter, [](scalar_t g, scalar_t z) -> scalar_t {
                return g * gamma_kernel(z) * digamma_kernel(z);
            });
        });

    return iter.output();
}

inline std::tuple<at::Tensor, at::Tensor> gamma_backward_backward(
    const at::Tensor& gg_input,
    const at::Tensor& grad,
    const at::Tensor& input
) {
    if (!gg_input.defined()) {
        return {at::Tensor(), at::Tensor()};
    }

    at::Tensor gg_output, new_grad;
    auto iter = at::TensorIteratorConfig()
        .add_output(gg_output)
        .add_output(new_grad)
        .add_const_input(gg_input)
        .add_const_input(grad)
        .add_const_input(input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf, iter.common_dtype(), "gamma_backward_backward_cpu", [&] {
            at::native::cpu_kernel_multiple_outputs(
                iter, [](scalar_t gg, scalar_t g, scalar_t z) -> std::tuple<scalar_t, scalar_t> {
                    scalar_t gamma_z = gamma_kernel(z);
                    scalar_t psi_z = digamma_kernel(z);
                    scalar_t psi1_z = trigamma_kernel(z);
                    scalar_t gg_out = gg * gamma_z * psi_z;
                    scalar_t new_grad = gg * g * gamma_z * (psi_z * psi_z + psi1_z);
                    return {gg_out, new_grad};
                });
        });

    return {iter.output(0), iter.output(1)};
}

}  // namespace torchscience::cpu

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("gamma", torchscience::cpu::gamma_forward);
    m.impl("gamma_backward", torchscience::cpu::gamma_backward);
    m.impl("gamma_backward_backward", torchscience::cpu::gamma_backward_backward);
}
