#include <gtest/gtest.h>
#include <torch/torch.h>

// Include the operator header to access meta kernel functions directly
#include "../../../../../src/torchscience/csrc/ops/example.h"

// Test fixture for meta kernel tests
class MetaExampleKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(42);
    }
};

// Test meta kernel for shape inference
TEST_F(MetaExampleKernelTest, ShapeInference) {
    auto input = torch::randn({10, 20, 30}, torch::kFloat32);

    // Test meta kernel
    auto result = science::ops::example_forward_meta(input, 1.0);

    // Should have same shape and dtype
    EXPECT_EQ(result.sizes(), input.sizes());
    EXPECT_EQ(result.dtype(), input.dtype());
}

TEST_F(MetaExampleKernelTest, PreservesDtype) {
    std::vector<torch::ScalarType> dtypes = {
        torch::kFloat32,
        torch::kFloat64,
        torch::kFloat16,
        torch::kInt32,
        torch::kInt64
    };

    for (auto dtype : dtypes) {
        auto input = torch::zeros({5, 5}, torch::TensorOptions().dtype(dtype));
        auto result = science::ops::example_forward_meta(input, 2.0);

        EXPECT_EQ(result.dtype(), dtype) << "Failed for dtype: " << dtype;
        EXPECT_EQ(result.sizes(), input.sizes());
    }
}

TEST_F(MetaExampleKernelTest, HandlesVariousShapes) {
    std::vector<std::vector<int64_t>> shapes = {
        {1},
        {10},
        {5, 5},
        {3, 4, 5},
        {2, 3, 4, 5},
        {0, 5},  // Empty tensor
        {1, 1, 1, 1}
    };

    for (const auto& shape : shapes) {
        auto input = torch::zeros(shape, torch::kFloat32);
        auto result = science::ops::example_forward_meta(input, 1.0);

        EXPECT_EQ(result.sizes(), input.sizes());
    }
}

TEST_F(MetaExampleKernelTest, BackwardShapeInference) {
    auto grad_out = torch::randn({10, 20}, torch::kFloat64);
    auto input = torch::randn({10, 20}, torch::kFloat64);

    // Test backward meta kernel
    auto grad_input = science::ops::example_backward_meta(grad_out, input, 1.0);

    // Gradient should have same shape as grad_out (and input)
    EXPECT_EQ(grad_input.sizes(), grad_out.sizes());
    EXPECT_EQ(grad_input.dtype(), grad_out.dtype());
}

TEST_F(MetaExampleKernelTest, NoDataAllocation) {
    auto input = torch::randn({1000, 1000}, torch::kFloat32);

    // Meta kernel should not actually allocate data or perform computation
    // It only computes metadata (shape, dtype, device)
    auto result = science::ops::example_forward_meta(input, 5.0);

    // Result has correct metadata
    EXPECT_EQ(result.sizes(), input.sizes());
    EXPECT_EQ(result.dtype(), input.dtype());

    // But the result is not actually computed (it's uninitialized)
    // We can't check the values as they're undefined
    // This is the expected behavior for meta kernels
}
