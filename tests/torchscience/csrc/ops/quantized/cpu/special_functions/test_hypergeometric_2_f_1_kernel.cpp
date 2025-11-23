#include "ops/special_functions.h"

#include <ATen/ATen.h>
#include <torch/torch.h>

#include <gtest/gtest.h>

#include <cmath>

// Test fixture for hypergeometric_2_f_1 Quantized CPU kernel tests
class Hypergeometric2F1QuantizedCPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        // No special setup needed for quantized CPU tests
    }

    // Helper to create quantized tensor
    at::Tensor create_quantized_tensor(const std::vector<double>& data,
                                        const std::vector<int64_t>& shape,
                                        double scale = 0.1,
                                        int64_t zero_point = 0) {
        auto float_tensor = torch::tensor(data, torch::kFloat32).reshape(shape);
        return torch::quantize_per_tensor(float_tensor, scale, zero_point, torch::kQInt8);
    }

    // Helper to compare with expected value within tolerance
    void AssertClose(double actual, double expected, double rtol = 1e-3, double atol = 1e-4) {
        double diff = std::abs(actual - expected);
        double tolerance = atol + rtol * std::abs(expected);
        ASSERT_LE(diff, tolerance)
            << "Values not close: actual=" << actual << ", expected=" << expected
            << ", diff=" << diff << ", tolerance=" << tolerance;
    }
};

TEST_F(Hypergeometric2F1QuantizedCPUTest, BasicFunctionality) {
    // Test basic functionality with simple quantized tensors
    auto a = create_quantized_tensor({1.0, 2.0, 3.0}, {3}, 0.1, 0);
    auto b = create_quantized_tensor({2.0, 3.0, 4.0}, {3}, 0.1, 0);
    auto c = create_quantized_tensor({3.0, 4.0, 5.0}, {3}, 0.1, 0);
    auto z = create_quantized_tensor({0.1, 0.2, 0.3}, {3}, 0.01, 0);

    auto result = science::ops::quantized::cpu::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.qscheme(), torch::kPerTensorAffine);
    EXPECT_TRUE(result.is_quantized());
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({3}));
}

TEST_F(Hypergeometric2F1QuantizedCPUTest, ScalePreservation) {
    // Test that scale is preserved during computation
    double scale = 0.05;
    int64_t zero_point = 10;

    auto a = create_quantized_tensor({1.0, 2.0, 3.0}, {3}, scale, zero_point);
    auto b = create_quantized_tensor({2.0, 3.0, 4.0}, {3}, scale, zero_point);
    auto c = create_quantized_tensor({3.0, 4.0, 5.0}, {3}, scale, zero_point);
    auto z = create_quantized_tensor({0.1, 0.2, 0.3}, {3}, scale, zero_point);

    auto result = science::ops::quantized::cpu::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    // Check that result has quantization parameters
    EXPECT_TRUE(result.is_quantized());
    EXPECT_GT(result.q_scale(), 0.0) << "Scale should be positive";
}

TEST_F(Hypergeometric2F1QuantizedCPUTest, ShapePreservation) {
    // Test that output shape matches input shape
    auto a = create_quantized_tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3}, 0.1, 0);
    auto b = create_quantized_tensor({2.0, 3.0, 4.0, 5.0, 6.0, 7.0}, {2, 3}, 0.1, 0);
    auto c = create_quantized_tensor({3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, {2, 3}, 0.1, 0);
    auto z = create_quantized_tensor({0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, {2, 3}, 0.01, 0);

    auto result = science::ops::quantized::cpu::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.sizes(), torch::IntArrayRef({2, 3}));
}

TEST_F(Hypergeometric2F1QuantizedCPUTest, EmptyTensor) {
    // Test with empty quantized tensors
    auto a = create_quantized_tensor({}, {0}, 0.1, 0);
    auto b = create_quantized_tensor({}, {0}, 0.1, 0);
    auto c = create_quantized_tensor({}, {0}, 0.1, 0);
    auto z = create_quantized_tensor({}, {0}, 0.1, 0);

    auto result = science::ops::quantized::cpu::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_EQ(result.numel(), 0);
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({0}));
}

TEST_F(Hypergeometric2F1QuantizedCPUTest, DifferentScales) {
    // Test with different quantization scales
    auto a = create_quantized_tensor({1.0, 2.0, 3.0}, {3}, 0.1, 0);
    auto b = create_quantized_tensor({2.0, 3.0, 4.0}, {3}, 0.2, 0);
    auto c = create_quantized_tensor({3.0, 4.0, 5.0}, {3}, 0.15, 0);
    auto z = create_quantized_tensor({0.1, 0.2, 0.3}, {3}, 0.05, 0);

    auto result = science::ops::quantized::cpu::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_TRUE(result.is_quantized());
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({3}));
}

TEST_F(Hypergeometric2F1QuantizedCPUTest, DifferentZeroPoints) {
    // Test with different zero points
    auto a = create_quantized_tensor({1.0, 2.0, 3.0}, {3}, 0.1, 0);
    auto b = create_quantized_tensor({2.0, 3.0, 4.0}, {3}, 0.1, 10);
    auto c = create_quantized_tensor({3.0, 4.0, 5.0}, {3}, 0.1, 20);
    auto z = create_quantized_tensor({0.1, 0.2, 0.3}, {3}, 0.1, -10);

    auto result = science::ops::quantized::cpu::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_TRUE(result.is_quantized());
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({3}));
}

TEST_F(Hypergeometric2F1QuantizedCPUTest, QuantizedFloatConsistency) {
    // Test that quantized and float implementations produce similar results
    // Create float tensors
    auto a_float = torch::tensor({1.0, 2.0, 3.0}, torch::kFloat32);
    auto b_float = torch::tensor({2.0, 3.0, 4.0}, torch::kFloat32);
    auto c_float = torch::tensor({3.0, 4.0, 5.0}, torch::kFloat32);
    auto z_float = torch::tensor({0.1, 0.2, 0.3}, torch::kFloat32);

    // Create quantized versions
    double scale = 0.05;
    auto a_quant = torch::quantize_per_tensor(a_float, scale, 0, torch::kQInt8);
    auto b_quant = torch::quantize_per_tensor(b_float, scale, 0, torch::kQInt8);
    auto c_quant = torch::quantize_per_tensor(c_float, scale, 0, torch::kQInt8);
    auto z_quant = torch::quantize_per_tensor(z_float, scale, 0, torch::kQInt8);

    auto result_float = science::ops::cpu::hypergeometric_2_f_1_forward_kernel(
        a_float.to(torch::kFloat64),
        b_float.to(torch::kFloat64),
        c_float.to(torch::kFloat64),
        z_float.to(torch::kFloat64)
    );

    auto result_quant = science::ops::quantized::cpu::hypergeometric_2_f_1_forward_kernel(
        a_quant, b_quant, c_quant, z_quant
    );

    // Dequantize for comparison
    auto result_quant_dequant = result_quant.dequantize();

    // Results should be similar (within quantization error)
    EXPECT_TRUE(torch::allclose(
        result_float.to(torch::kFloat32),
        result_quant_dequant,
        /*rtol=*/0.1,  // Relaxed tolerance for quantization
        /*atol=*/0.5
    )) << "Quantized and float results differ significantly";
}


TEST_F(Hypergeometric2F1QuantizedCPUTest, NumericalStability) {
    // Test numerical stability with various quantized input values
    auto test_cases = std::vector<std::vector<double>>{
        {1.0, 2.0, 3.0, 0.1},
        {0.5, 1.5, 2.5, 0.5},
        {2.0, 3.0, 4.0, 0.2},
        {1.5, 2.5, 3.5, -0.3}
    };

    for (const auto& test_case : test_cases) {
        auto a = create_quantized_tensor({test_case[0]}, {1}, 0.1, 0);
        auto b = create_quantized_tensor({test_case[1]}, {1}, 0.1, 0);
        auto c = create_quantized_tensor({test_case[2]}, {1}, 0.1, 0);
        auto z = create_quantized_tensor({test_case[3]}, {1}, 0.01, 0);

        auto result = science::ops::quantized::cpu::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

        // Dequantize and check finiteness
        auto result_dequant = result.dequantize();
        EXPECT_TRUE(torch::isfinite(result_dequant).all().item<bool>())
            << "Result not finite for test case";
    }
}

