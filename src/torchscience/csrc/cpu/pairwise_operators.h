#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::cpu {

// =============================================================================
// CPUPairwiseOperator - Template for pairwise distance/similarity operators
// =============================================================================

// PairwiseTraits must provide:
//   - template<T> static T compute(const T* x, const T* y, int64_t d, Args... args);
//   - template<T> static void backward(T grad, const T* x, const T* y, int64_t d, T* grad_x, T* grad_y, Args...);

template<typename PairwiseTraits>
struct CPUPairwiseOperator {
    template<typename... Args>
    static at::Tensor forward(
        const at::Tensor& x,  // (m, d)
        const at::Tensor& y,  // (n, d)
        Args... args
    ) {
        TORCH_CHECK(x.dim() == 2, "pairwise: x must be 2D (m, d)");
        TORCH_CHECK(y.dim() == 2, "pairwise: y must be 2D (n, d)");
        TORCH_CHECK(x.size(1) == y.size(1), "pairwise: feature dimensions must match");

        int64_t m = x.size(0);
        int64_t n = y.size(0);
        int64_t d = x.size(1);

        at::Tensor x_contig = x.contiguous();
        at::Tensor y_contig = y.contiguous();
        at::Tensor output = at::empty({m, n}, x.options());

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            x.scalar_type(),
            "pairwise_cpu",
            [&]() {
                const scalar_t* x_ptr = x_contig.data_ptr<scalar_t>();
                const scalar_t* y_ptr = y_contig.data_ptr<scalar_t>();
                scalar_t* out_ptr = output.data_ptr<scalar_t>();

                at::parallel_for(0, m * n, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t idx = begin; idx < end; ++idx) {
                        int64_t i = idx / n;
                        int64_t j = idx % n;
                        out_ptr[idx] = PairwiseTraits::template compute<scalar_t>(
                            x_ptr + i * d,
                            y_ptr + j * d,
                            d,
                            args...
                        );
                    }
                });
            }
        );

        return output;
    }

    template<typename... Args>
    static std::tuple<at::Tensor, at::Tensor> backward(
        const at::Tensor& grad_output,  // (m, n)
        const at::Tensor& x,  // (m, d)
        const at::Tensor& y,  // (n, d)
        Args... args
    ) {
        int64_t m = x.size(0);
        int64_t n = y.size(0);
        int64_t d = x.size(1);

        at::Tensor x_contig = x.contiguous();
        at::Tensor y_contig = y.contiguous();
        at::Tensor grad_contig = grad_output.contiguous();

        at::Tensor grad_x = at::zeros_like(x);
        at::Tensor grad_y = at::zeros_like(y);

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            x.scalar_type(),
            "pairwise_backward_cpu",
            [&]() {
                const scalar_t* grad_ptr = grad_contig.data_ptr<scalar_t>();
                const scalar_t* x_ptr = x_contig.data_ptr<scalar_t>();
                const scalar_t* y_ptr = y_contig.data_ptr<scalar_t>();
                scalar_t* grad_x_ptr = grad_x.data_ptr<scalar_t>();
                scalar_t* grad_y_ptr = grad_y.data_ptr<scalar_t>();

                // Accumulate gradients (note: requires atomic or per-thread buffers for y)
                // Simplified: sequential accumulation for correctness
                for (int64_t i = 0; i < m; ++i) {
                    for (int64_t j = 0; j < n; ++j) {
                        scalar_t grad_val = grad_ptr[i * n + j];
                        std::vector<scalar_t> temp_grad_x(d, scalar_t(0));
                        std::vector<scalar_t> temp_grad_y(d, scalar_t(0));

                        PairwiseTraits::template backward<scalar_t>(
                            grad_val,
                            x_ptr + i * d,
                            y_ptr + j * d,
                            d,
                            temp_grad_x.data(),
                            temp_grad_y.data(),
                            args...
                        );

                        for (int64_t k = 0; k < d; ++k) {
                            grad_x_ptr[i * d + k] += temp_grad_x[k];
                            grad_y_ptr[j * d + k] += temp_grad_y[k];
                        }
                    }
                }
            }
        );

        return std::make_tuple(grad_x, grad_y);
    }
};

}  // namespace torchscience::cpu
