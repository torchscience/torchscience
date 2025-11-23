#include "ops/special_functions.h"

#include <ATen/ATen.h>
#include <torch/torch.h>

#include <gtest/gtest.h>

#include <cmath>

// Test fixture for hypergeometric_2_f_1 CPU kernel tests
class Hypergeometric2F1CPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up can be used for common initialization
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

TEST_F(Hypergeometric2F1CPUTest, BasicFunctionality) {
    // Test basic functionality with simple scalar inputs
    auto a = torch::tensor({1.0}, torch::kFloat64);
    auto b = torch::tensor({2.0}, torch::kFloat64);
    auto c = torch::tensor({3.0}, torch::kFloat64);
    auto z = torch::tensor({0.5}, torch::kFloat64);

    auto result = science::ops::cpu::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.sizes(), torch::IntArrayRef({1}));
    EXPECT_EQ(result.scalar_type(), torch::kFloat64);
    EXPECT_TRUE(result.device().is_cpu());
}

TEST_F(Hypergeometric2F1CPUTest, SpecialValueZeroZ) {
    // Test that ₂F₁(a,b;c;0) = 1
    auto a = torch::rand({5}, torch::kFloat64);
    auto b = torch::rand({5}, torch::kFloat64);
    auto c = torch::rand({5}, torch::kFloat64) + 1.0;
    auto z = torch::zeros({5}, torch::kFloat64);

    auto result = science::ops::cpu::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    auto expected = torch::ones({5}, torch::kFloat64);
    EXPECT_TRUE(torch::allclose(result, expected, /*rtol=*/1e-10, /*atol=*/1e-12));
}

TEST_F(Hypergeometric2F1CPUTest, ShapePreservation) {
    // Test that output shape matches broadcasted input shape
    auto a = torch::randn({10, 5}, torch::kFloat64);
    auto b = torch::randn({10, 5}, torch::kFloat64);
    auto c = torch::randn({10, 5}, torch::kFloat64) + 1.0;
    auto z = torch::randn({10, 5}, torch::kFloat64) * 0.5;

    auto result = science::ops::cpu::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.sizes(), torch::IntArrayRef({10, 5}));
}

TEST_F(Hypergeometric2F1CPUTest, Broadcasting) {
    // Test broadcasting behavior
    auto a = torch::randn({5, 1}, torch::kFloat64);
    auto b = torch::randn({1, 3}, torch::kFloat64);
    auto c = torch::randn({5, 3}, torch::kFloat64) + 1.0;
    auto z = torch::randn({5, 3}, torch::kFloat64) * 0.5;

    auto result = science::ops::cpu::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.sizes(), torch::IntArrayRef({5, 3}));
}

TEST_F(Hypergeometric2F1CPUTest, EmptyTensor) {
    // Test with empty tensors
    auto a = torch::empty({0}, torch::kFloat64);
    auto b = torch::empty({0}, torch::kFloat64);
    auto c = torch::empty({0}, torch::kFloat64);
    auto z = torch::empty({0}, torch::kFloat64);

    auto result = science::ops::cpu::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.numel(), 0);
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({0}));
}

TEST_F(Hypergeometric2F1CPUTest, Float32Dtype) {
    // Test with float32 dtype
    auto a = torch::randn({10}, torch::kFloat32);
    auto b = torch::randn({10}, torch::kFloat32);
    auto c = torch::randn({10}, torch::kFloat32) + 1.0f;
    auto z = torch::randn({10}, torch::kFloat32) * 0.5f;

    auto result = science::ops::cpu::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.scalar_type(), torch::kFloat32);
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({10}));
}

TEST_F(Hypergeometric2F1CPUTest, ComplexFloat64) {
    // Test with complex128 dtype
    auto a = torch::randn({5}, torch::kComplexDouble);
    auto b = torch::randn({5}, torch::kComplexDouble);
    auto c = torch::randn({5}, torch::kComplexDouble) + 1.0;
    auto z = torch::randn({5}, torch::kComplexDouble) * 0.3;

    auto result = science::ops::cpu::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.scalar_type(), torch::kComplexDouble);
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({5}));
}

TEST_F(Hypergeometric2F1CPUTest, NonContiguous) {
    // Test with non-contiguous tensors (transposed)
    auto a = torch::randn({10, 20}, torch::kFloat64).t();
    auto b = torch::randn({10, 20}, torch::kFloat64).t();
    auto c = torch::randn({10, 20}, torch::kFloat64).t() + 1.0;
    auto z = torch::randn({10, 20}, torch::kFloat64).t() * 0.5;

    EXPECT_FALSE(a.is_contiguous());

    auto result = science::ops::cpu::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.sizes(), torch::IntArrayRef({20, 10}));
}

TEST_F(Hypergeometric2F1CPUTest, BackwardShapeCorrectness) {
    // Test backward pass returns correctly shaped gradients
    auto grad_out = torch::randn({5}, torch::kFloat64);
    auto a = torch::randn({5}, torch::kFloat64);
    auto b = torch::randn({5}, torch::kFloat64);
    auto c = torch::randn({5}, torch::kFloat64) + 1.0;
    auto z = torch::randn({5}, torch::kFloat64) * 0.5;
    auto result = torch::randn({5}, torch::kFloat64);  // Placeholder for result

    auto grads = science::ops::cpu::hypergeometric_2_f_1_backward_kernel(
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

TEST_F(Hypergeometric2F1CPUTest, NumericalStability) {
    // Test numerical stability with various input ranges
    auto test_values = {0.1, 0.5, 0.9, -0.5, -1.0};

    for (auto z_val : test_values) {
        auto a = torch::tensor({1.0}, torch::kFloat64);
        auto b = torch::tensor({2.0}, torch::kFloat64);
        auto c = torch::tensor({3.0}, torch::kFloat64);
        auto z = torch::tensor({z_val}, torch::kFloat64);

        auto result = science::ops::cpu::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

        // Result should be finite
        EXPECT_TRUE(torch::isfinite(result).all().item<bool>())
            << "Result not finite for z=" << z_val;
    }
}

