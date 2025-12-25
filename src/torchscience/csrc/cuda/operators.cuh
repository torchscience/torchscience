#pragma once

#include <tuple>

#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/TensorIterator.h>
#include <thrust/tuple.h>
#include <torch/library.h>

namespace torchscience::cuda {

// =============================================================================
// CUDAUnaryOperator - Template for unary operators (mirrors CPUUnaryOperator)
// =============================================================================

template<typename ImplTraits>
struct CUDAUnaryOperator {
    static at::Tensor forward(const at::Tensor& input) {
        at::Tensor output;

        auto iter = at::TensorIteratorConfig()
            .add_output(output)
            .add_const_input(input)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build();

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            iter.common_dtype(),
            "cuda_unary_forward",
            [&]() {
                at::native::gpu_kernel(iter, []GPU_LAMBDA(scalar_t x) -> scalar_t {
                    return ImplTraits::template forward<scalar_t>(x);
                });
            }
        );

        return iter.output();
    }

    static at::Tensor backward(
        const at::Tensor& grad_output,
        const at::Tensor& input
    ) {
        at::Tensor grad_input;

        auto iter = at::TensorIteratorConfig()
            .add_output(grad_input)
            .add_const_input(grad_output)
            .add_const_input(input)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build();

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            iter.common_dtype(),
            "cuda_unary_backward",
            [&]() {
                at::native::gpu_kernel(iter, []GPU_LAMBDA(scalar_t g, scalar_t x) -> scalar_t {
                    return ImplTraits::template backward<scalar_t>(g, x);
                });
            }
        );

        return iter.output();
    }

    static std::tuple<at::Tensor, at::Tensor> backward_backward(
        const at::Tensor& grad_grad_input,
        const at::Tensor& grad_output,
        const at::Tensor& input
    ) {
        const bool has_gg = grad_grad_input.defined();

        if (!has_gg) {
            return std::make_tuple(at::Tensor(), at::Tensor());
        }

        at::Tensor grad_grad_output;
        at::Tensor grad_input;

        auto iter = at::TensorIteratorConfig()
            .add_output(grad_grad_output)
            .add_output(grad_input)
            .add_const_input(grad_grad_input)
            .add_const_input(grad_output)
            .add_const_input(input)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build();

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            iter.common_dtype(),
            "cuda_unary_backward_backward",
            [&]() {
                at::native::gpu_kernel_multiple_outputs(
                    iter,
                    [has_gg]GPU_LAMBDA(scalar_t gg, scalar_t g, scalar_t x)
                        -> thrust::tuple<scalar_t, scalar_t> {
                        auto [gg_out, new_grad] = ImplTraits::template backward_backward<scalar_t>(
                            gg, g, x, has_gg
                        );
                        return thrust::make_tuple(gg_out, new_grad);
                    }
                );
            }
        );

        return std::make_tuple(iter.output(0), iter.output(1));
    }

    static void register_all(
        torch::Library& module,
        const char* name,
        const char* backward_name,
        const char* backward_backward_name
    ) {
        module.impl(name, &forward);
        module.impl(backward_name, &backward);
        module.impl(backward_backward_name, &backward_backward);
    }
};

#define REGISTER_CUDA_UNARY(module, name, Impl) \
    ::torchscience::cuda::CUDAUnaryOperator<Impl>::register_all( \
        module, #name, #name "_backward", #name "_backward_backward")

// =============================================================================
// CUDABinaryOperator - Template for binary operators
// =============================================================================

template<typename ImplTraits>
struct CUDABinaryOperator {
    static at::Tensor forward(
        const at::Tensor& input1,
        const at::Tensor& input2
    ) {
        at::Tensor output;

        auto iter = at::TensorIteratorConfig()
            .add_output(output)
            .add_const_input(input1)
            .add_const_input(input2)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build();

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            iter.common_dtype(),
            "cuda_binary_forward",
            [&]() {
                at::native::gpu_kernel(iter, []GPU_LAMBDA(scalar_t x1, scalar_t x2) -> scalar_t {
                    return ImplTraits::template forward<scalar_t>(x1, x2);
                });
            }
        );

        return iter.output();
    }

    static std::tuple<at::Tensor, at::Tensor> backward(
        const at::Tensor& grad_output,
        const at::Tensor& input1,
        const at::Tensor& input2
    ) {
        at::Tensor grad_input1;
        at::Tensor grad_input2;

        auto iter = at::TensorIteratorConfig()
            .add_output(grad_input1)
            .add_output(grad_input2)
            .add_const_input(grad_output)
            .add_const_input(input1)
            .add_const_input(input2)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build();

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            iter.common_dtype(),
            "cuda_binary_backward",
            [&]() {
                at::native::gpu_kernel_multiple_outputs(
                    iter,
                    []GPU_LAMBDA(scalar_t g, scalar_t x1, scalar_t x2)
                        -> thrust::tuple<scalar_t, scalar_t> {
                        auto [g1, g2] = ImplTraits::template backward<scalar_t>(g, x1, x2);
                        return thrust::make_tuple(g1, g2);
                    }
                );
            }
        );

        return std::make_tuple(iter.output(0), iter.output(1));
    }

    static std::tuple<at::Tensor, at::Tensor, at::Tensor> backward_backward(
        const at::Tensor& grad_grad_input1,
        const at::Tensor& grad_grad_input2,
        const at::Tensor& grad_output,
        const at::Tensor& input1,
        const at::Tensor& input2
    ) {
        const bool has_gg1 = grad_grad_input1.defined();
        const bool has_gg2 = grad_grad_input2.defined();

        if (!has_gg1 && !has_gg2) {
            return std::make_tuple(at::Tensor(), at::Tensor(), at::Tensor());
        }

        at::Tensor grad_grad_output;
        at::Tensor grad_input1;
        at::Tensor grad_input2;

        at::Tensor gg1_input = has_gg1 ? grad_grad_input1 : at::zeros_like(grad_output);
        at::Tensor gg2_input = has_gg2 ? grad_grad_input2 : at::zeros_like(grad_output);

        auto iter = at::TensorIteratorConfig()
            .add_output(grad_grad_output)
            .add_output(grad_input1)
            .add_output(grad_input2)
            .add_const_input(gg1_input)
            .add_const_input(gg2_input)
            .add_const_input(grad_output)
            .add_const_input(input1)
            .add_const_input(input2)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build();

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            iter.common_dtype(),
            "cuda_binary_backward_backward",
            [&]() {
                at::native::gpu_kernel_multiple_outputs(
                    iter,
                    [has_gg1, has_gg2]GPU_LAMBDA(
                        scalar_t gg1, scalar_t gg2, scalar_t g, scalar_t x1, scalar_t x2
                    ) -> thrust::tuple<scalar_t, scalar_t, scalar_t> {
                        auto [gg_out, g1, g2] = ImplTraits::template backward_backward<scalar_t>(
                            gg1, gg2, g, x1, x2, has_gg1, has_gg2
                        );
                        return thrust::make_tuple(gg_out, g1, g2);
                    }
                );
            }
        );

        return std::make_tuple(iter.output(0), iter.output(1), iter.output(2));
    }

    static void register_all(
        torch::Library& module,
        const char* name,
        const char* backward_name,
        const char* backward_backward_name
    ) {
        module.impl(name, &forward);
        module.impl(backward_name, &backward);
        module.impl(backward_backward_name, &backward_backward);
    }
};

#define REGISTER_CUDA_BINARY(module, name, Impl) \
    ::torchscience::cuda::CUDABinaryOperator<Impl>::register_all( \
        module, #name, #name "_backward", #name "_backward_backward")

}  // namespace torchscience::cuda
