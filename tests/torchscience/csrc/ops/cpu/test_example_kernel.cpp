#include <gtest/gtest.h>
#include <torch/torch.h>

// Include the operator header to access CPU kernel functions directly
#include "../../../../../src/torchscience/csrc/ops/example.h"

// Test fixture for CPU kernel tests
class CPUExampleKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Seed for reproducibility
        torch::manual_seed(42);
    }
};

// Basic functionality tests
TEST_F(CPUExampleKernelTest, BasicOperation) {
    auto input = torch::randn({5, 5}, torch::kFloat32);
    double scalar = 2.0;

    // Call the CPU kernel directly
    auto result = science::ops::cpu::example_forward_kernel(input, scalar);

    // Verify result
    EXPECT_EQ(result.sizes(), input.sizes());
    EXPECT_EQ(result.dtype(), input.dtype());
    EXPECT_TRUE(result.device().is_cpu());

    // Verify mathematical correctness
    auto expected = input + scalar;
    EXPECT_TRUE(torch::allclose(result, expected));
}

TEST_F(CPUExampleKernelTest, HandlesEmptyTensor) {
    auto input = torch::empty({0, 5}, torch::kFloat32);
    auto result = science::ops::cpu::example_forward_kernel(input, 2.0);

    EXPECT_EQ(result.sizes(), input.sizes());
    EXPECT_EQ(result.numel(), 0);
}

TEST_F(CPUExampleKernelTest, HandlesScalarTensor) {
    auto input = torch::tensor(3.14f);
    auto result = science::ops::cpu::example_forward_kernel(input, 1.0);

    EXPECT_EQ(result.numel(), 1);
    EXPECT_FLOAT_EQ(result.item<float>(), 4.14f);
}

TEST_F(CPUExampleKernelTest, PreservesNonContiguous) {
    auto input = torch::randn({10, 10}, torch::kFloat32).transpose(0, 1);
    EXPECT_FALSE(input.is_contiguous());

    auto result = science::ops::cpu::example_forward_kernel(input, 1.5);

    // Result should have same shape
    EXPECT_EQ(result.sizes(), input.sizes());

    // Verify correctness regardless of contiguity
    auto expected = input + 1.5;
    EXPECT_TRUE(torch::allclose(result, expected));
}

// Backward pass tests
TEST_F(CPUExampleKernelTest, BackwardPassCorrectness) {
    auto input = torch::randn({5, 5}, torch::kFloat64);
    auto grad_output = torch::ones({5, 5}, torch::kFloat64);

    // Test backward kernel directly
    auto grad_input = science::ops::cpu::example_backward_kernel(grad_output, input, 2.0);

    // Gradient of (input + 2.0) with respect to input is 1
    // So grad_input should equal grad_output
    EXPECT_TRUE(torch::allclose(grad_input, grad_output));
}

// Dtype preservation tests using parameterized testing
class CPUExampleDtypeTest : public ::testing::TestWithParam<torch::ScalarType> {};

TEST_P(CPUExampleDtypeTest, PreservesDtype) {
    auto dtype = GetParam();
    auto input = torch::randn({5, 5}, torch::TensorOptions().dtype(dtype));
    auto result = science::ops::cpu::example_forward_kernel(input, 2.0);

    EXPECT_EQ(result.dtype(), dtype);
    EXPECT_EQ(result.sizes(), input.sizes());
}

INSTANTIATE_TEST_SUITE_P(
    AllDtypes,
    CPUExampleDtypeTest,
    ::testing::Values(
        torch::kFloat32,
        torch::kFloat64,
        torch::kFloat16
    )
);

// Shape tests using parameterized testing
class CPUExampleShapeTest : public ::testing::TestWithParam<std::vector<int64_t>> {};

TEST_P(CPUExampleShapeTest, HandlesVariousShapes) {
    auto shape = GetParam();
    auto input = torch::randn(shape, torch::kFloat32);
    auto result = science::ops::cpu::example_forward_kernel(input, 1.0);

    EXPECT_EQ(result.sizes(), input.sizes());

    // Verify correctness
    auto expected = input + 1.0;
    EXPECT_TRUE(torch::allclose(result, expected));
}

INSTANTIATE_TEST_SUITE_P(
    VariousShapes,
    CPUExampleShapeTest,
    ::testing::Values(
        std::vector<int64_t>{10},           // 1D
        std::vector<int64_t>{5, 5},         // 2D
        std::vector<int64_t>{3, 4, 5},      // 3D
        std::vector<int64_t>{2, 3, 4, 5},   // 4D
        std::vector<int64_t>{1, 1, 1, 1}    // Edge case: all dimensions 1
    )
);

// Scalar value tests
TEST_F(CPUExampleKernelTest, VariousScalarValues) {
    auto input = torch::ones({3, 3}, torch::kFloat32);

    std::vector<double> scalar_values = {0.0, 1.0, -1.0, 3.14, -2.718, 1e6, -1e6};

    for (double scalar : scalar_values) {
        auto result = science::ops::cpu::example_forward_kernel(input, scalar);
        auto expected = input + scalar;

        EXPECT_TRUE(torch::allclose(result, expected))
            << "Failed for scalar value: " << scalar;
    }
}

// Edge case: very large tensors (stress test)
TEST_F(CPUExampleKernelTest, HandlesLargeTensors) {
    auto input = torch::randn({1000, 1000}, torch::kFloat32);
    auto result = science::ops::cpu::example_forward_kernel(input, 1.0);

    EXPECT_EQ(result.sizes(), input.sizes());

    auto expected = input + 1.0;
    EXPECT_TRUE(torch::allclose(result, expected));
}

// Test that CPU tensor constraint is enforced
TEST_F(CPUExampleKernelTest, CPUTensorConstraint) {
    auto input = torch::randn({5, 5}, torch::kFloat32);

    // This should succeed (CPU tensor)
    EXPECT_NO_THROW({
        auto result = science::ops::cpu::example_forward_kernel(input, 1.0);
    });
}

// Numerical precision test
TEST_F(CPUExampleKernelTest, NumericalPrecision) {
    // Use double precision for high accuracy
    auto input = torch::randn({100, 100}, torch::kFloat64);
    double scalar = 1.23456789012345;

    auto result = science::ops::cpu::example_forward_kernel(input, scalar);
    auto expected = input + scalar;

    // Check with very tight tolerance for float64
    EXPECT_TRUE(torch::allclose(result, expected, /*rtol=*/1e-12, /*atol=*/1e-14));
}
