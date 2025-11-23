#include "ops/special_functions.h"

#include <ATen/ATen.h>
#include <torch/torch.h>

#include <gtest/gtest.h>

#include <cmath>

// Test fixture for hypergeometric_2_f_1 Meta kernel tests
class Hypergeometric2F1MetaTest : public ::testing::Test {
protected:
    void SetUp() override {
        // No special setup needed for meta tests
    }

    // Helper to create meta tensor (fake tensor without data)
    at::Tensor create_meta_tensor(const std::vector<int64_t>& shape,
                                    at::ScalarType dtype = torch::kFloat64) {
        return at::empty(shape, at::TensorOptions().dtype(dtype).device(at::kMeta));
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

TEST_F(Hypergeometric2F1MetaTest, BasicMetaFunctionality) {
    // Test basic functionality with meta tensors
    auto a = create_meta_tensor({5});
    auto b = create_meta_tensor({5});
    auto c = create_meta_tensor({5});
    auto z = create_meta_tensor({5});

    auto result = science::ops::hypergeometric_2_f_1_forward_meta(a, b, c, z);

    // Check that result is a meta tensor with correct shape
    EXPECT_TRUE(result.is_meta());
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({5}));
    EXPECT_EQ(result.scalar_type(), torch::kFloat64);
}

TEST_F(Hypergeometric2F1MetaTest, ShapeInference) {
    // Test that meta kernel correctly infers output shape
    auto a = create_meta_tensor({10, 5});
    auto b = create_meta_tensor({10, 5});
    auto c = create_meta_tensor({10, 5});
    auto z = create_meta_tensor({10, 5});

    auto result = science::ops::hypergeometric_2_f_1_forward_meta(a, b, c, z);

    EXPECT_TRUE(result.is_meta());
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({10, 5}));
}

TEST_F(Hypergeometric2F1MetaTest, Broadcasting) {
    // Test broadcasting with meta tensors
    auto a = create_meta_tensor({5, 1});
    auto b = create_meta_tensor({1, 3});
    auto c = create_meta_tensor({5, 3});
    auto z = create_meta_tensor({5, 3});

    auto result = science::ops::hypergeometric_2_f_1_forward_meta(a, b, c, z);

    EXPECT_TRUE(result.is_meta());
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({5, 3}));
}

TEST_F(Hypergeometric2F1MetaTest, BroadcastingAllInputs) {
    // Test broadcasting when all inputs have different shapes
    auto a = create_meta_tensor({1});
    auto b = create_meta_tensor({5, 1});
    auto c = create_meta_tensor({3, 1});
    auto z = create_meta_tensor({5, 3});

    auto result = science::ops::hypergeometric_2_f_1_forward_meta(a, b, c, z);

    EXPECT_TRUE(result.is_meta());
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({5, 3}));
}

TEST_F(Hypergeometric2F1MetaTest, EmptyTensor) {
    // Test with empty meta tensors
    auto a = create_meta_tensor({0});
    auto b = create_meta_tensor({0});
    auto c = create_meta_tensor({0});
    auto z = create_meta_tensor({0});

    auto result = science::ops::hypergeometric_2_f_1_forward_meta(a, b, c, z);

    EXPECT_TRUE(result.is_meta());
    EXPECT_EQ(result.numel(), 0);
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({0}));
}

TEST_F(Hypergeometric2F1MetaTest, ScalarTensor) {
    // Test with scalar (0-dim) tensors
    auto a = create_meta_tensor({});
    auto b = create_meta_tensor({});
    auto c = create_meta_tensor({});
    auto z = create_meta_tensor({});

    auto result = science::ops::hypergeometric_2_f_1_forward_meta(a, b, c, z);

    EXPECT_TRUE(result.is_meta());
    EXPECT_EQ(result.dim(), 0);
}

TEST_F(Hypergeometric2F1MetaTest, Float32Dtype) {
    // Test with float32 dtype
    auto a = create_meta_tensor({10}, torch::kFloat32);
    auto b = create_meta_tensor({10}, torch::kFloat32);
    auto c = create_meta_tensor({10}, torch::kFloat32);
    auto z = create_meta_tensor({10}, torch::kFloat32);

    auto result = science::ops::hypergeometric_2_f_1_forward_meta(a, b, c, z);

    EXPECT_TRUE(result.is_meta());
    EXPECT_EQ(result.scalar_type(), torch::kFloat32);
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({10}));
}

TEST_F(Hypergeometric2F1MetaTest, ComplexFloat64) {
    // Test with complex128 dtype
    auto a = create_meta_tensor({5}, torch::kComplexDouble);
    auto b = create_meta_tensor({5}, torch::kComplexDouble);
    auto c = create_meta_tensor({5}, torch::kComplexDouble);
    auto z = create_meta_tensor({5}, torch::kComplexDouble);

    auto result = science::ops::hypergeometric_2_f_1_forward_meta(a, b, c, z);

    EXPECT_TRUE(result.is_meta());
    EXPECT_EQ(result.scalar_type(), torch::kComplexDouble);
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({5}));
}

TEST_F(Hypergeometric2F1MetaTest, ComplexFloat32) {
    // Test with complex64 dtype
    auto a = create_meta_tensor({5}, torch::kComplexFloat);
    auto b = create_meta_tensor({5}, torch::kComplexFloat);
    auto c = create_meta_tensor({5}, torch::kComplexFloat);
    auto z = create_meta_tensor({5}, torch::kComplexFloat);

    auto result = science::ops::hypergeometric_2_f_1_forward_meta(a, b, c, z);

    EXPECT_TRUE(result.is_meta());
    EXPECT_EQ(result.scalar_type(), torch::kComplexFloat);
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({5}));
}

TEST_F(Hypergeometric2F1MetaTest, MixedDtypes) {
    // Test with mixed dtypes (should promote)
    auto a = create_meta_tensor({5}, torch::kFloat32);
    auto b = create_meta_tensor({5}, torch::kFloat32);
    auto c = create_meta_tensor({5}, torch::kFloat64);
    auto z = create_meta_tensor({5}, torch::kFloat32);

    auto result = science::ops::hypergeometric_2_f_1_forward_meta(a, b, c, z);

    EXPECT_TRUE(result.is_meta());
    // Result should be promoted to float64
    EXPECT_EQ(result.scalar_type(), torch::kFloat64);
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({5}));
}

TEST_F(Hypergeometric2F1MetaTest, HighDimensionalTensors) {
    // Test with high-dimensional tensors
    auto a = create_meta_tensor({2, 3, 4, 5});
    auto b = create_meta_tensor({2, 3, 4, 5});
    auto c = create_meta_tensor({2, 3, 4, 5});
    auto z = create_meta_tensor({2, 3, 4, 5});

    auto result = science::ops::hypergeometric_2_f_1_forward_meta(a, b, c, z);

    EXPECT_TRUE(result.is_meta());
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({2, 3, 4, 5}));
}

TEST_F(Hypergeometric2F1MetaTest, BroadcastingWithScalars) {
    // Test broadcasting with scalar and multi-dimensional tensors
    auto a = create_meta_tensor({});  // Scalar
    auto b = create_meta_tensor({10, 5});
    auto c = create_meta_tensor({1, 5});
    auto z = create_meta_tensor({10, 1});

    auto result = science::ops::hypergeometric_2_f_1_forward_meta(a, b, c, z);

    EXPECT_TRUE(result.is_meta());
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({10, 5}));
}

TEST_F(Hypergeometric2F1MetaTest, NoDataAllocation) {
    // Verify that meta kernel doesn't allocate actual data
    auto a = create_meta_tensor({1000, 1000});
    auto b = create_meta_tensor({1000, 1000});
    auto c = create_meta_tensor({1000, 1000});
    auto z = create_meta_tensor({1000, 1000});

    auto result = science::ops::hypergeometric_2_f_1_forward_meta(a, b, c, z);

    // Meta tensor should have no storage
    EXPECT_TRUE(result.is_meta());
    EXPECT_FALSE(result.has_storage());
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({1000, 1000}));
}

TEST_F(Hypergeometric2F1MetaTest, DtypePromotionRealComplex) {
    // Test dtype promotion between real and complex
    auto a = create_meta_tensor({5}, torch::kFloat64);
    auto b = create_meta_tensor({5}, torch::kComplexDouble);
    auto c = create_meta_tensor({5}, torch::kFloat64);
    auto z = create_meta_tensor({5}, torch::kFloat64);

    auto result = science::ops::hypergeometric_2_f_1_forward_meta(a, b, c, z);

    EXPECT_TRUE(result.is_meta());
    // Should promote to complex
    EXPECT_EQ(result.scalar_type(), torch::kComplexDouble);
}

TEST_F(Hypergeometric2F1MetaTest, ConsistencyWithCPUShape) {
    // Test that meta and CPU kernels produce same shape
    // Create regular CPU tensors
    auto a_cpu = torch::randn({5, 3}, torch::kFloat64);
    auto b_cpu = torch::randn({1, 3}, torch::kFloat64);
    auto c_cpu = torch::randn({5, 1}, torch::kFloat64);
    auto z_cpu = torch::randn({1, 1}, torch::kFloat64);

    // Create equivalent meta tensors
    auto a_meta = create_meta_tensor({5, 3});
    auto b_meta = create_meta_tensor({1, 3});
    auto c_meta = create_meta_tensor({5, 1});
    auto z_meta = create_meta_tensor({1, 1});

    auto result_cpu = science::ops::cpu::hypergeometric_2_f_1_forward_kernel(a_cpu, b_cpu, c_cpu, z_cpu);
    auto result_meta = science::ops::hypergeometric_2_f_1_forward_meta(a_meta, b_meta, c_meta, z_meta);

    // Shapes should match
    EXPECT_EQ(result_cpu.sizes(), result_meta.sizes());
    EXPECT_EQ(result_cpu.scalar_type(), result_meta.scalar_type());
}

