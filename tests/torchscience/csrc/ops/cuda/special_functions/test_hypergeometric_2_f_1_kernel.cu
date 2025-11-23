#include <gtest/gtest.h>

#include "ops/special_functions.h"

#include <ATen/ATen.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>

#include <cmath>

// Test fixture for hypergeometric_2_f_1 CUDA kernel tests
class Hypergeometric2F1CUDATest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if CUDA is available
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available, skipping CUDA tests";
        }
    }

    // Helper to compare with expected value within tolerance
    void AssertClose(double actual, double expected, double rtol = 1e-5, double atol = 1e-7) {
        double diff = std::abs(actual - expected);
        double tolerance = atol + rtol * std::abs(expected);
        ASSERT_LE(diff, tolerance)
            << "Values not close: actual=" << actual << ", expected=" << expected
            << ", diff=" << diff << ", tolerance=" << tolerance;
    }
};

TEST_F(Hypergeometric2F1CUDATest, BasicFunctionality) {
    // Test basic functionality with simple scalar inputs on CUDA
    auto a = torch::tensor({1.0}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    auto b = torch::tensor({2.0}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    auto c = torch::tensor({3.0}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    auto z = torch::tensor({0.5}, torch::dtype(torch::kFloat64).device(torch::kCUDA));

    auto result = science::ops::cuda::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.sizes(), torch::IntArrayRef({1}));
    EXPECT_EQ(result.scalar_type(), torch::kFloat64);
    EXPECT_TRUE(result.device().is_cuda());
}

TEST_F(Hypergeometric2F1CUDATest, SpecialValueZeroZ) {
    // Test that ₂F₁(a,b;c;0) = 1
    auto a = torch::rand({5}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    auto b = torch::rand({5}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    auto c = torch::rand({5}, torch::dtype(torch::kFloat64).device(torch::kCUDA)) + 1.0;
    auto z = torch::zeros({5}, torch::dtype(torch::kFloat64).device(torch::kCUDA));

    auto result = science::ops::cuda::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    auto expected = torch::ones({5}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    EXPECT_TRUE(torch::allclose(result, expected, /*rtol=*/1e-10, /*atol=*/1e-12));
}

TEST_F(Hypergeometric2F1CUDATest, ShapePreservation) {
    // Test that output shape matches broadcasted input shape
    auto a = torch::randn({10, 5}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    auto b = torch::randn({10, 5}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    auto c = torch::randn({10, 5}, torch::dtype(torch::kFloat64).device(torch::kCUDA)) + 1.0;
    auto z = torch::randn({10, 5}, torch::dtype(torch::kFloat64).device(torch::kCUDA)) * 0.5;

    auto result = science::ops::cuda::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.sizes(), torch::IntArrayRef({10, 5}));
}

TEST_F(Hypergeometric2F1CUDATest, Broadcasting) {
    // Test broadcasting behavior
    auto a = torch::randn({5, 1}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    auto b = torch::randn({1, 3}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    auto c = torch::randn({5, 3}, torch::dtype(torch::kFloat64).device(torch::kCUDA)) + 1.0;
    auto z = torch::randn({5, 3}, torch::dtype(torch::kFloat64).device(torch::kCUDA)) * 0.5;

    auto result = science::ops::cuda::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.sizes(), torch::IntArrayRef({5, 3}));
}

TEST_F(Hypergeometric2F1CUDATest, EmptyTensor) {
    // Test with empty tensors
    auto a = torch::empty({0}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    auto b = torch::empty({0}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    auto c = torch::empty({0}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    auto z = torch::empty({0}, torch::dtype(torch::kFloat64).device(torch::kCUDA));

    auto result = science::ops::cuda::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.numel(), 0);
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({0}));
}

TEST_F(Hypergeometric2F1CUDATest, Float32Dtype) {
    // Test with float32 dtype
    auto a = torch::randn({10}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto b = torch::randn({10}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto c = torch::randn({10}, torch::dtype(torch::kFloat32).device(torch::kCUDA)) + 1.0f;
    auto z = torch::randn({10}, torch::dtype(torch::kFloat32).device(torch::kCUDA)) * 0.5f;

    auto result = science::ops::cuda::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.scalar_type(), torch::kFloat32);
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({10}));
}

TEST_F(Hypergeometric2F1CUDATest, ComplexFloat64) {
    // Test with complex128 dtype
    auto a = torch::randn({5}, torch::dtype(torch::kComplexDouble).device(torch::kCUDA));
    auto b = torch::randn({5}, torch::dtype(torch::kComplexDouble).device(torch::kCUDA));
    auto c = torch::randn({5}, torch::dtype(torch::kComplexDouble).device(torch::kCUDA)) + 1.0;
    auto z = torch::randn({5}, torch::dtype(torch::kComplexDouble).device(torch::kCUDA)) * 0.3;

    auto result = science::ops::cuda::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.scalar_type(), torch::kComplexDouble);
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({5}));
}

TEST_F(Hypergeometric2F1CUDATest, NonContiguous) {
    // Test with non-contiguous tensors (transposed)
    auto a = torch::randn({10, 20}, torch::dtype(torch::kFloat64).device(torch::kCUDA)).t();
    auto b = torch::randn({10, 20}, torch::dtype(torch::kFloat64).device(torch::kCUDA)).t();
    auto c = torch::randn({10, 20}, torch::dtype(torch::kFloat64).device(torch::kCUDA)).t() + 1.0;
    auto z = torch::randn({10, 20}, torch::dtype(torch::kFloat64).device(torch::kCUDA)).t() * 0.5;

    EXPECT_FALSE(a.is_contiguous());

    auto result = science::ops::cuda::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.sizes(), torch::IntArrayRef({20, 10}));
}

TEST_F(Hypergeometric2F1CUDATest, BackwardShapeCorrectness) {
    // Test backward pass returns correctly shaped gradients
    auto grad_out = torch::randn({5}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    auto a = torch::randn({5}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    auto b = torch::randn({5}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    auto c = torch::randn({5}, torch::dtype(torch::kFloat64).device(torch::kCUDA)) + 1.0;
    auto z = torch::randn({5}, torch::dtype(torch::kFloat64).device(torch::kCUDA)) * 0.5;
    auto result = torch::randn({5}, torch::dtype(torch::kFloat64).device(torch::kCUDA));  // Placeholder for result

    auto grads = science::ops::cuda::hypergeometric_2_f_1_backward_kernel(
        grad_out, a, b, c, z, result);

    auto grad_a = std::get<0>(grads);
    auto grad_b = std::get<1>(grads);
    auto grad_c = std::get<2>(grads);
    auto grad_z = std::get<3>(grads);

    EXPECT_EQ(grad_a.sizes(), a.sizes());
    EXPECT_EQ(grad_b.sizes(), b.sizes());
    EXPECT_EQ(grad_c.sizes(), c.sizes());
    EXPECT_EQ(grad_z.sizes(), z.sizes());
}

TEST_F(Hypergeometric2F1CUDATest, CPUCUDAConsistency) {
    // Test that CPU and CUDA implementations produce consistent results
    auto a_cpu = torch::randn({10}, torch::kFloat64);
    auto b_cpu = torch::randn({10}, torch::kFloat64);
    auto c_cpu = torch::randn({10}, torch::kFloat64) + 1.0;
    auto z_cpu = torch::randn({10}, torch::kFloat64) * 0.5;

    auto a_cuda = a_cpu.cuda();
    auto b_cuda = b_cpu.cuda();
    auto c_cuda = c_cpu.cuda();
    auto z_cuda = z_cpu.cuda();

    auto result_cpu = science::ops::cpu::hypergeometric_2_f_1_forward_kernel(a_cpu, b_cpu, c_cpu, z_cpu);
    auto result_cuda = science::ops::cuda::hypergeometric_2_f_1_forward_kernel(a_cuda, b_cuda, c_cuda, z_cuda);

    // Move CUDA result back to CPU for comparison
    auto result_cuda_cpu = result_cuda.cpu();

    EXPECT_TRUE(torch::allclose(result_cpu, result_cuda_cpu, /*rtol=*/1e-10, /*atol=*/1e-12))
        << "CPU and CUDA results differ";
}

TEST_F(Hypergeometric2F1CUDATest, LargeTensorStressTest) {
    // Test with large tensors to stress GPU parallelization
    auto a = torch::randn({1000, 1000}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto b = torch::randn({1000, 1000}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto c = torch::randn({1000, 1000}, torch::dtype(torch::kFloat32).device(torch::kCUDA)) + 1.0f;
    auto z = torch::randn({1000, 1000}, torch::dtype(torch::kFloat32).device(torch::kCUDA)) * 0.5f;

    auto result = science::ops::cuda::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.sizes(), torch::IntArrayRef({1000, 1000}));
    EXPECT_TRUE(torch::isfinite(result).all().item<bool>())
        << "Large tensor computation produced non-finite values";
}

TEST_F(Hypergeometric2F1CUDATest, NumericalStability) {
    // Test numerical stability with various input ranges
    auto test_values = {0.1, 0.5, 0.9, -0.5, -1.0};

    for (auto z_val : test_values) {
        auto a = torch::tensor({1.0}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
        auto b = torch::tensor({2.0}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
        auto c = torch::tensor({3.0}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
        auto z = torch::tensor({z_val}, torch::dtype(torch::kFloat64).device(torch::kCUDA));

        auto result = science::ops::cuda::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

        // Result should be finite
        EXPECT_TRUE(torch::isfinite(result).all().item<bool>())
            << "Result not finite for z=" << z_val;
    }
}
