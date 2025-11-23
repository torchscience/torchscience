#include <gtest/gtest.h>

#include "ops/special_functions.h"

#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <torch/torch.h>

#include <cmath>

// Test fixture for hypergeometric_2_f_1 Autocast kernel tests
class Hypergeometric2F1AutocastTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if CUDA is available for autocast testing
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available, skipping autocast tests (requires CUDA)";
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

TEST_F(Hypergeometric2F1AutocastTest, BasicAutocastBehavior) {
    // Test that autocast correctly promotes types
    auto a = torch::randn({5}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto b = torch::randn({5}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto c = torch::randn({5}, torch::dtype(torch::kFloat32).device(torch::kCUDA)) + 1.0f;
    auto z = torch::randn({5}, torch::dtype(torch::kFloat32).device(torch::kCUDA)) * 0.5f;

    // Enable autocast mode
    at::autocast::set_autocast_gpu_dtype(at::kHalf);
    at::autocast::set_autocast_enabled(true);

    auto result = science::ops::autocast::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    at::autocast::set_autocast_enabled(false);

    // Result should exist and be valid
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({5}));
    EXPECT_TRUE(result.device().is_cuda());
}

TEST_F(Hypergeometric2F1AutocastTest, Float32Input) {
    // Test autocast with float32 input tensors
    auto a = torch::randn({10}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto b = torch::randn({10}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto c = torch::randn({10}, torch::dtype(torch::kFloat32).device(torch::kCUDA)) + 1.0f;
    auto z = torch::randn({10}, torch::dtype(torch::kFloat32).device(torch::kCUDA)) * 0.5f;

    auto result = science::ops::autocast::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.sizes(), torch::IntArrayRef({10}));
}

TEST_F(Hypergeometric2F1AutocastTest, Float64Input) {
    // Test autocast with float64 input tensors (should not be modified)
    auto a = torch::randn({10}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    auto b = torch::randn({10}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    auto c = torch::randn({10}, torch::dtype(torch::kFloat64).device(torch::kCUDA)) + 1.0;
    auto z = torch::randn({10}, torch::dtype(torch::kFloat64).device(torch::kCUDA)) * 0.5;

    auto result = science::ops::autocast::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    // Float64 should pass through unchanged
    EXPECT_EQ(result.scalar_type(), torch::kFloat64);
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({10}));
}

TEST_F(Hypergeometric2F1AutocastTest, MixedPrecisionInputs) {
    // Test autocast with mixed precision inputs
    auto a = torch::randn({10}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto b = torch::randn({10}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto c = torch::randn({10}, torch::dtype(torch::kFloat64).device(torch::kCUDA)) + 1.0;
    auto z = torch::randn({10}, torch::dtype(torch::kFloat32).device(torch::kCUDA)) * 0.5f;

    auto result = science::ops::autocast::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.sizes(), torch::IntArrayRef({10}));
}

TEST_F(Hypergeometric2F1AutocastTest, ShapePreservation) {
    // Test that autocast preserves shapes
    auto a = torch::randn({10, 5}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto b = torch::randn({10, 5}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto c = torch::randn({10, 5}, torch::dtype(torch::kFloat32).device(torch::kCUDA)) + 1.0f;
    auto z = torch::randn({10, 5}, torch::dtype(torch::kFloat32).device(torch::kCUDA)) * 0.5f;

    auto result = science::ops::autocast::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.sizes(), torch::IntArrayRef({10, 5}));
}

TEST_F(Hypergeometric2F1AutocastTest, Broadcasting) {
    // Test that autocast works with broadcasting
    auto a = torch::randn({5, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto b = torch::randn({1, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto c = torch::randn({5, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA)) + 1.0f;
    auto z = torch::randn({5, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA)) * 0.5f;

    auto result = science::ops::autocast::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.sizes(), torch::IntArrayRef({5, 3}));
}

TEST_F(Hypergeometric2F1AutocastTest, EmptyTensor) {
    // Test autocast with empty tensors
    auto a = torch::empty({0}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto b = torch::empty({0}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto c = torch::empty({0}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto z = torch::empty({0}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    auto result = science::ops::autocast::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.numel(), 0);
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({0}));
}

TEST_F(Hypergeometric2F1AutocastTest, AutocastEnabledVsDisabled) {
    // Test behavior with autocast enabled vs disabled
    auto a = torch::randn({10}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto b = torch::randn({10}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto c = torch::randn({10}, torch::dtype(torch::kFloat32).device(torch::kCUDA)) + 1.0f;
    auto z = torch::randn({10}, torch::dtype(torch::kFloat32).device(torch::kCUDA)) * 0.5f;

    // Without autocast
    at::autocast::set_autocast_enabled(false);
    auto result_no_autocast = science::ops::autocast::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    // With autocast
    at::autocast::set_autocast_gpu_dtype(at::kHalf);
    at::autocast::set_autocast_enabled(true);
    auto result_with_autocast = science::ops::autocast::hypergeometric_2_f_1_forward_kernel(a, b, c, z);
    at::autocast::set_autocast_enabled(false);

    // Both should produce valid results
    EXPECT_EQ(result_no_autocast.sizes(), result_with_autocast.sizes());
}

TEST_F(Hypergeometric2F1AutocastTest, BackwardShapeCorrectness) {
    // Test backward pass with autocast
    auto grad_out = torch::randn({5}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto a = torch::randn({5}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto b = torch::randn({5}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto c = torch::randn({5}, torch::dtype(torch::kFloat32).device(torch::kCUDA)) + 1.0f;
    auto z = torch::randn({5}, torch::dtype(torch::kFloat32).device(torch::kCUDA)) * 0.5f;
    auto result = torch::randn({5}, torch::dtype(torch::kFloat32).device(torch::kCUDA));  // Placeholder

    auto grads = science::ops::autocast::hypergeometric_2_f_1_backward_kernel(
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

TEST_F(Hypergeometric2F1AutocastTest, NumericalStability) {
    // Test numerical stability with autocast
    auto test_values = {0.1, 0.5, 0.9, -0.5, -1.0};

    for (auto z_val : test_values) {
        auto a = torch::tensor({1.0}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        auto b = torch::tensor({2.0}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        auto c = torch::tensor({3.0}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        auto z = torch::tensor({z_val}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

        auto result = science::ops::autocast::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

        // Result should be finite
        EXPECT_TRUE(torch::isfinite(result).all().item<bool>())
            << "Result not finite for z=" << z_val;
    }
}

