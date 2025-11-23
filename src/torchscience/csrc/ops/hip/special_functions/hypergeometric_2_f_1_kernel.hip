#include "../../hypergeometric_2_f_1.h"

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <c10/util/complex.h>
#include <torch/library.h>

namespace science {
namespace ops {
namespace hip {

// Device function to compute hypergeometric function for a single element
template <typename scalar_t>
__device__ scalar_t hypergeometric_2_f_1_device_impl(
    scalar_t a, scalar_t b, scalar_t c, scalar_t z) {

    // Helper functions for complex absolute value
    auto abs_val = [](const scalar_t& x) {
        if constexpr (c10::is_complex<scalar_t>::value) {
            return std::sqrt(x.real() * x.real() + x.imag() * x.imag());
        } else {
            return std::abs(x);
        }
    };

    // Helper for real part
    auto real_part = [](const scalar_t& x) {
        if constexpr (c10::is_complex<scalar_t>::value) {
            return x.real();
        } else {
            return x;
        }
    };

    // Handle special cases
    if (abs_val(z) < scalar_t(1e-14)) {
        return scalar_t(1.0);
    }

    // For |z| >= 1, use Pfaff transformation if appropriate
    if (abs_val(z) >= scalar_t(1.0) && real_part(z) > scalar_t(0.5)) {
        scalar_t one_minus_z = scalar_t(1.0) - z;
        scalar_t factor = scalar_t(1.0);
        for (int i = 0; i < 10; ++i) {
            factor = factor / one_minus_z;
        }
        scalar_t z_trans = z / (z - scalar_t(1.0));
        return factor * hypergeometric_2_f_1_device_impl(a, c - b, c, z_trans);
    }

    // Series expansion
    constexpr int max_iterations = 1000;
    constexpr double tolerance = 1e-10;

    scalar_t sum = scalar_t(1.0);
    scalar_t term = scalar_t(1.0);

    for (int n = 0; n < max_iterations; ++n) {
        scalar_t a_n = a + scalar_t(n);
        scalar_t b_n = b + scalar_t(n);
        scalar_t c_n = c + scalar_t(n);
        scalar_t n_plus_1 = scalar_t(n + 1);

        term = term * (a_n * b_n * z) / (c_n * n_plus_1);
        sum = sum + term;

        if (abs_val(term) < tolerance * abs_val(sum)) {
            break;
        }
    }

    return sum;
}

// HIP kernel for forward pass
template <typename scalar_t>
__global__ void hypergeometric_2_f_1_hip_kernel_forward(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ c,
    const scalar_t* __restrict__ z,
    scalar_t* __restrict__ output,
    int64_t numel) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int64_t i = idx; i < numel; i += blockDim.x * gridDim.x) {
        output[i] = hypergeometric_2_f_1_device_impl(a[i], b[i], c[i], z[i]);
    }
}

// Helper to get optimal HIP launch configuration
int get_num_threads() {
    return 256;
}

int get_num_blocks(int64_t numel, int threads_per_block) {
    const int max_blocks = 65535;
    return std::min(max_blocks,
                    static_cast<int>((numel + threads_per_block - 1) / threads_per_block));
}

// Forward pass implementation
at::Tensor hypergeometric_2_f_1_forward_kernel(
    const at::Tensor& a, const at::Tensor& b,
    const at::Tensor& c, const at::Tensor& z) {

    TORCH_CHECK(a.device().is_hip(), "a must be a HIP tensor");
    TORCH_CHECK(b.device().is_hip(), "b must be a HIP tensor");
    TORCH_CHECK(c.device().is_hip(), "c must be a HIP tensor");
    TORCH_CHECK(z.device().is_hip(), "z must be a HIP tensor");

    c10::hip::HIPGuard device_guard(z.device());

    // Broadcast all inputs together
    auto a_b = at::broadcast_tensors({a, b, c, z});
    auto a_broad = a_b[0].contiguous();
    auto b_broad = a_b[1].contiguous();
    auto c_broad = a_b[2].contiguous();
    auto z_broad = a_b[3].contiguous();

    auto result = at::empty_like(z_broad);

    int64_t numel = result.numel();

    if (numel == 0) {
        return result;
    }

    const int threads = get_num_threads();
    const int blocks = get_num_blocks(numel, threads);

    // Dispatch kernel based on dtype
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
        result.scalar_type(), "hypergeometric_2_f_1_hip_forward", [&] {
            hipLaunchKernelGGL(
                hypergeometric_2_f_1_hip_kernel_forward<scalar_t>,
                dim3(blocks), dim3(threads), 0, at::hip::getCurrentHIPStream(),
                a_broad.data_ptr<scalar_t>(),
                b_broad.data_ptr<scalar_t>(),
                c_broad.data_ptr<scalar_t>(),
                z_broad.data_ptr<scalar_t>(),
                result.data_ptr<scalar_t>(),
                numel);
        });

    C10_HIP_KERNEL_LAUNCH_CHECK();

    return result;
}

// Backward pass implementation
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
hypergeometric_2_f_1_backward_kernel(
    const at::Tensor& grad_out,
    const at::Tensor& a, const at::Tensor& b,
    const at::Tensor& c, const at::Tensor& z,
    const at::Tensor& result) {

    c10::hip::HIPGuard device_guard(grad_out.device());

    // Gradient w.r.t. z
    auto a_plus_1 = a + 1.0;
    auto b_plus_1 = b + 1.0;
    auto c_plus_1 = c + 1.0;
    auto hyp_shifted = hypergeometric_2_f_1_forward_kernel(a_plus_1, b_plus_1, c_plus_1, z);
    auto grad_z = grad_out * (a * b / c) * hyp_shifted;

    // Gradients w.r.t. a, b, c using finite differences
    const double eps = 1e-4;

    auto a_plus_eps = a + eps;
    auto f_a_plus = hypergeometric_2_f_1_forward_kernel(a_plus_eps, b, c, z);
    auto grad_a = grad_out * (f_a_plus - result) / eps;

    auto b_plus_eps = b + eps;
    auto f_b_plus = hypergeometric_2_f_1_forward_kernel(a, b_plus_eps, c, z);
    auto grad_b = grad_out * (f_b_plus - result) / eps;

    auto c_plus_eps = c + eps;
    auto f_c_plus = hypergeometric_2_f_1_forward_kernel(a, b, c_plus_eps, z);
    auto grad_c = grad_out * (f_c_plus - result) / eps;

    return std::make_tuple(grad_a, grad_b, grad_c, grad_z);
}

}  // namespace hip
}  // namespace ops
}  // namespace science

// Register HIP implementations
TORCH_LIBRARY_IMPL(torchscience, HIP, module) {
    module.impl(TORCH_SELECTIVE_NAME("torchscience::hypergeometric_2_f_1"),
                TORCH_FN(science::ops::hip::hypergeometric_2_f_1_forward_kernel));

    module.impl(TORCH_SELECTIVE_NAME("torchscience::_hypergeometric_2_f_1_backward"),
                TORCH_FN(science::ops::hip::hypergeometric_2_f_1_backward_kernel));
}
