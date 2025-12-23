#pragma once

#include <vector>
#include <ATen/Dispatch.h>
#include <ATen/Generator.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/TensorIterator.h>
#include <torch/library.h>
#include "../core/creation_common.h"

namespace torchscience::cpu {

// =============================================================================
// CPUCreationOperator - Deterministic creation operators
// =============================================================================

// CreationTraits must provide:
//   - static std::vector<int64_t> output_shape(params...);
//   - template<typename scalar_t> static void kernel(scalar_t* out, int64_t numel, params...);

template<typename CreationTraits>
struct CPUCreationOperator {
    template<typename... Args>
    static at::Tensor forward(
        Args... args,
        const c10::optional<at::ScalarType>& dtype,
        const c10::optional<at::Layout>& layout,
        const c10::optional<at::Device>& device,
        bool requires_grad
    ) {
        std::vector<int64_t> shape_vec = CreationTraits::output_shape(args...);
        ::torchscience::core::check_size_nonnegative(shape_vec, "creation_op");

        auto options = ::torchscience::core::build_options(dtype, layout, device, at::kCPU);
        int64_t numel = ::torchscience::core::compute_numel(shape_vec);

        at::Tensor output = at::empty(shape_vec, options);

        if (numel > 0) {
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kBFloat16,
                at::kHalf,
                output.scalar_type(),
                "cpu_creation",
                [&]() {
                    CreationTraits::template kernel<scalar_t>(
                        output.data_ptr<scalar_t>(),
                        numel,
                        args...
                    );
                }
            );
        }

        if (requires_grad) {
            output = output.requires_grad_(true);
        }

        return output;
    }
};

#define REGISTER_CPU_CREATION(module, name, Traits, ...) \
    module.impl(#name, &::torchscience::cpu::CPUCreationOperator<Traits>::forward<__VA_ARGS__>)

// =============================================================================
// CPUStochasticCreationOperator - Creation operators with RNG
// =============================================================================

// StochasticCreationTraits must provide:
//   - static std::vector<int64_t> output_shape(params...);
//   - template<typename scalar_t, typename RNG>
//     static void kernel(scalar_t* out, int64_t numel, RNG* rng, params...);

template<typename CreationTraits>
struct CPUStochasticCreationOperator {
    template<typename... Args>
    static at::Tensor forward(
        Args... args,
        c10::optional<at::Generator> generator,
        const c10::optional<at::ScalarType>& dtype,
        const c10::optional<at::Layout>& layout,
        const c10::optional<at::Device>& device,
        bool requires_grad
    ) {
        std::vector<int64_t> shape_vec = CreationTraits::output_shape(args...);
        ::torchscience::core::check_size_nonnegative(shape_vec, "stochastic_creation_op");

        auto options = ::torchscience::core::build_options(dtype, layout, device, at::kCPU);
        int64_t numel = ::torchscience::core::compute_numel(shape_vec);

        at::Tensor output = at::empty(shape_vec, options);

        if (numel > 0) {
            auto gen = at::get_generator_or_default<at::CPUGeneratorImpl>(
                generator, at::detail::getDefaultCPUGenerator()
            );
            std::lock_guard<std::mutex> lock(gen->mutex_);

            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kBFloat16,
                at::kHalf,
                output.scalar_type(),
                "cpu_stochastic_creation",
                [&]() {
                    CreationTraits::template kernel<scalar_t>(
                        output.data_ptr<scalar_t>(),
                        numel,
                        gen,
                        args...
                    );
                }
            );
        }

        if (requires_grad) {
            output = output.requires_grad_(true);
        }

        return output;
    }
};

#define REGISTER_CPU_STOCHASTIC_CREATION(module, name, Traits, ...) \
    module.impl(#name, &::torchscience::cpu::CPUStochasticCreationOperator<Traits>::forward<__VA_ARGS__>)

}  // namespace torchscience::cpu
