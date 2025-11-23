#include <gtest/gtest.h>

#include "ops/special_functions.h"

#include <ATen/ATen.h>
#include <torch/torch.h>

#include <cmath>

// Test fixture for hypergeometric_2_f_1 MPS kernel tests
class Hypergeometric2F1MPSTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if MPS is available (only on macOS with Apple Silicon)
        if (!torch::hasMPS()) {
            GTEST_SKIP() << "MPS not available, skipping MPS tests";
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

TEST_F(Hypergeometric2F1MPSTest, BasicFunctionality) {
    // Test basic functionality with simple scalar inputs on MPS
    auto a = torch::tensor({1.0}, torch::dtype(torch::kFloat64).device(torch::kMPS));
    auto b = torch::tensor({2.0}, torch::dtype(torch::kFloat64).device(torch::kMPS));
    auto c = torch::tensor({3.0}, torch::dtype(torch::kFloat64).device(torch::kMPS));
    auto z = torch::tensor({0.5}, torch::dtype(torch::kFloat64).device(torch::kMPS));

    auto result = science::ops::mps::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.sizes(), torch::IntArrayRef({1}));
    EXPECT_EQ(result.scalar_type(), torch::kFloat64);
    EXPECT_TRUE(result.device().is_mps());
}

TEST_F(Hypergeometric2F1MPSTest, SpecialValueZeroZ) {
    // Test that ₂F₁(a,b;c;0) = 1
    auto a = torch::rand({5}, torch::dtype(torch::kFloat64).device(torch::kMPS));
    auto b = torch::rand({5}, torch::dtype(torch::kFloat64).device(torch::kMPS));
    auto c = torch::rand({5}, torch::dtype(torch::kFloat64).device(torch::kMPS)) + 1.0;
    auto z = torch::zeros({5}, torch::dtype(torch::kFloat64).device(torch::kMPS));

    auto result = science::ops::mps::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    auto expected = torch::ones({5}, torch::dtype(torch::kFloat64).device(torch::kMPS));
    EXPECT_TRUE(torch::allclose(result, expected, /*rtol=*/1e-10, /*atol=*/1e-12));
}

TEST_F(Hypergeometric2F1MPSTest, ShapePreservation) {
    // Test that output shape matches broadcasted input shape
    auto a = torch::randn({10, 5}, torch::dtype(torch::kFloat64).device(torch::kMPS));
    auto b = torch::randn({10, 5}, torch::dtype(torch::kFloat64).device(torch::kMPS));
    auto c = torch::randn({10, 5}, torch::dtype(torch::kFloat64).device(torch::kMPS)) + 1.0;
    auto z = torch::randn({10, 5}, torch::dtype(torch::kFloat64).device(torch::kMPS)) * 0.5;

    auto result = science::ops::mps::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.sizes(), torch::IntArrayRef({10, 5}));
}

TEST_F(Hypergeometric2F1MPSTest, Broadcasting) {
    // Test broadcasting behavior
    auto a = torch::randn({5, 1}, torch::dtype(torch::kFloat64).device(torch::kMPS));
    auto b = torch::randn({1, 3}, torch::dtype(torch::kFloat64).device(torch::kMPS));
    auto c = torch::randn({5, 3}, torch::dtype(torch::kFloat64).device(torch::kMPS)) + 1.0;
    auto z = torch::randn({5, 3}, torch::dtype(torch::kFloat64).device(torch::kMPS)) * 0.5;

    auto result = science::ops::mps::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.sizes(), torch::IntArrayRef({5, 3}));
}

TEST_F(Hypergeometric2F1MPSTest, EmptyTensor) {
    // Test with empty tensors
    auto a = torch::empty({0}, torch::dtype(torch::kFloat64).device(torch::kMPS));
    auto b = torch::empty({0}, torch::dtype(torch::kFloat64).device(torch::kMPS));
    auto c = torch::empty({0}, torch::dtype(torch::kFloat64).device(torch::kMPS));
    auto z = torch::empty({0}, torch::dtype(torch::kFloat64).device(torch::kMPS));

    auto result = science::ops::mps::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.numel(), 0);
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({0}));
}

TEST_F(Hypergeometric2F1MPSTest, Float32Dtype) {
    // Test with float32 dtype
    auto a = torch::randn({10}, torch::dtype(torch::kFloat32).device(torch::kMPS));
    auto b = torch::randn({10}, torch::dtype(torch::kFloat32).device(torch::kMPS));
    auto c = torch::randn({10}, torch::dtype(torch::kFloat32).device(torch::kMPS)) + 1.0f;
    auto z = torch::randn({10}, torch::dtype(torch::kFloat32).device(torch::kMPS)) * 0.5f;

    auto result = science::ops::mps::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.scalar_type(), torch::kFloat32);
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({10}));
}

TEST_F(Hypergeometric2F1MPSTest, ComplexFloat64) {
    // Test with complex128 dtype
    auto a = torch::randn({5}, torch::dtype(torch::kComplexDouble).device(torch::kMPS));
    auto b = torch::randn({5}, torch::dtype(torch::kComplexDouble).device(torch::kMPS));
    auto c = torch::randn({5}, torch::dtype(torch::kComplexDouble).device(torch::kMPS)) + 1.0;
    auto z = torch::randn({5}, torch::dtype(torch::kComplexDouble).device(torch::kMPS)) * 0.3;

    auto result = science::ops::mps::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.scalar_type(), torch::kComplexDouble);
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({5}));
}

TEST_F(Hypergeometric2F1MPSTest, NonContiguous) {
    // Test with non-contiguous tensors (transposed)
    auto a = torch::randn({10, 20}, torch::dtype(torch::kFloat64).device(torch::kMPS)).t();
    auto b = torch::randn({10, 20}, torch::dtype(torch::kFloat64).device(torch::kMPS)).t();
    auto c = torch::randn({10, 20}, torch::dtype(torch::kFloat64).device(torch::kMPS)).t() + 1.0;
    auto z = torch::randn({10, 20}, torch::dtype(torch::kFloat64).device(torch::kMPS)).t() * 0.5;

    EXPECT_FALSE(a.is_contiguous());

    auto result = science::ops::mps::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.sizes(), torch::IntArrayRef({20, 10}));
}

TEST_F(Hypergeometric2F1MPSTest, BackwardShapeCorrectness) {
    // Test backward pass returns correctly shaped gradients
    auto grad_out = torch::randn({5}, torch::dtype(torch::kFloat64).device(torch::kMPS));
    auto a = torch::randn({5}, torch::dtype(torch::kFloat64).device(torch::kMPS));
    auto b = torch::randn({5}, torch::dtype(torch::kFloat64).device(torch::kMPS));
    auto c = torch::randn({5}, torch::dtype(torch::kFloat64).device(torch::kMPS)) + 1.0;
    auto z = torch::randn({5}, torch::dtype(torch::kFloat64).device(torch::kMPS)) * 0.5;
    auto result = torch::randn({5}, torch::dtype(torch::kFloat64).device(torch::kMPS));  // Placeholder for result

    auto grads = science::ops::mps::hypergeometric_2_f_1_backward_kernel(
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

TEST_F(Hypergeometric2F1MPSTest, CPUMPSConsistency) {
    // Test that CPU and MPS implementations produce consistent results
    // MPS uses CPU fallback, so results should be identical
    auto a_cpu = torch::randn({10}, torch::kFloat64);
    auto b_cpu = torch::randn({10}, torch::kFloat64);
    auto c_cpu = torch::randn({10}, torch::kFloat64) + 1.0;
    auto z_cpu = torch::randn({10}, torch::kFloat64) * 0.5;

    auto a_mps = a_cpu.to(torch::kMPS);
    auto b_mps = b_cpu.to(torch::kMPS);
    auto c_mps = c_cpu.to(torch::kMPS);
    auto z_mps = z_cpu.to(torch::kMPS);

    auto result_cpu = science::ops::cpu::hypergeometric_2_f_1_forward_kernel(a_cpu, b_cpu, c_cpu, z_cpu);
    auto result_mps = science::ops::mps::hypergeometric_2_f_1_forward_kernel(a_mps, b_mps, c_mps, z_mps);

    // Move MPS result back to CPU for comparison
    auto result_mps_cpu = result_mps.cpu();

    EXPECT_TRUE(torch::allclose(result_cpu, result_mps_cpu, /*rtol=*/1e-10, /*atol=*/1e-12))
        << "CPU and MPS results differ";
}

TEST_F(Hypergeometric2F1MPSTest, NumericalStability) {
    // Test numerical stability with various input ranges
    auto test_values = {0.1, 0.5, 0.9, -0.5, -1.0};

    for (auto z_val : test_values) {
        auto a = torch::tensor({1.0}, torch::dtype(torch::kFloat64).device(torch::kMPS));
        auto b = torch::tensor({2.0}, torch::dtype(torch::kFloat64).device(torch::kMPS));
        auto c = torch::tensor({3.0}, torch::dtype(torch::kFloat64).device(torch::kMPS));
        auto z = torch::tensor({z_val}, torch::dtype(torch::kFloat64).device(torch::kMPS));

        auto result = science::ops::mps::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

        // Result should be finite
        EXPECT_TRUE(torch::isfinite(result).all().item<bool>())
            << "Result not finite for z=" << z_val;
    }
}

