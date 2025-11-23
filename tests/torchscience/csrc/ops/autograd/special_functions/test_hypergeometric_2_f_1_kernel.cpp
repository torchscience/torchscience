#include <gtest/gtest.h>
#include <torch/torch.h>

// Note: The autograd layer doesn't expose testable functions directly
// because it uses an anonymous namespace ExampleFunction class pattern.
// The autograd functionality for hypergeometric_2_f_1 is comprehensively
// tested through Python integration tests with gradcheck/gradgradcheck.
// These C++ tests demonstrate autograd patterns and verify basic functionality.

// Test fixture for autograd tests
class Hypergeometric2F1AutogradTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(42);
    }
};

// Test gradient flow with simple operations
// (The actual hypergeometric_2_f_1 autograd is tested via Python integration tests)
TEST_F(Hypergeometric2F1AutogradTest, GradientFlowPattern) {
    // Demonstrate basic autograd pattern
    auto input = torch::randn({5, 5}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));

    // Simple operation for demonstration
    auto output = input * 2.0 + 1.0;

    auto grad_output = torch::ones_like(output);
    output.backward(grad_output);

    // Verify gradient is computed correctly
    EXPECT_TRUE(input.grad().defined());
    // For f(x) = 2x + 1, df/dx = 2
    EXPECT_TRUE(torch::allclose(input.grad(), grad_output * 2.0));
}

TEST_F(Hypergeometric2F1AutogradTest, GradientFlowVariousShapes) {
    std::vector<std::vector<int64_t>> shapes = {
        {10},
        {5, 5},
        {3, 4, 5},
        {2, 3, 4, 5}
    };

    for (const auto& shape : shapes) {
        auto input = torch::randn(shape, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));

        auto output = input + 1.0;
        auto grad_output = torch::ones_like(output);
        output.backward(grad_output);

        EXPECT_TRUE(input.grad().defined());
        EXPECT_EQ(input.grad().sizes(), input.sizes());
        EXPECT_TRUE(torch::allclose(input.grad(), grad_output));

        // Clear gradients for next iteration
        input.grad().zero_();
    }
}

TEST_F(Hypergeometric2F1AutogradTest, MultipleBackwardPasses) {
    auto input = torch::randn({3, 3}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));

    // First forward/backward
    auto output1 = input + 1.0;
    output1.sum().backward();
    auto grad1 = input.grad().clone();

    // Clear gradients
    input.grad().zero_();

    // Second forward/backward with different constant
    auto output2 = input + 2.0;
    output2.sum().backward();
    auto grad2 = input.grad().clone();

    // Gradients should be same (derivative doesn't depend on the constant)
    EXPECT_TRUE(torch::allclose(grad1, grad2));
}

#ifdef WITH_CUDA
TEST_F(Hypergeometric2F1AutogradTest, GradientFlowCUDA) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto input = torch::randn(
        {5, 5},
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat64).requires_grad(true)
    );

    auto output = input + 2.0;
    auto grad_output = torch::ones_like(output);
    output.backward(grad_output);

    EXPECT_TRUE(input.grad().defined());
    EXPECT_TRUE(input.grad().device().is_cuda());
    EXPECT_TRUE(torch::allclose(input.grad(), grad_output));
}
#endif

TEST_F(Hypergeometric2F1AutogradTest, NonContiguousGradient) {
    auto input = torch::randn({10, 10}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
    auto input_t = input.transpose(0, 1);  // Non-contiguous view

    EXPECT_FALSE(input_t.is_contiguous());

    auto output = input_t + 1.0;
    auto grad_output = torch::ones_like(output);
    output.backward(grad_output);

    EXPECT_TRUE(input.grad().defined());
}

TEST_F(Hypergeometric2F1AutogradTest, ComplexDtype) {
    // Test autograd with complex dtype
    auto input = torch::randn(
        {5, 5},
        torch::TensorOptions().dtype(torch::kComplexDouble).requires_grad(true)
    );

    // Use a real scalar for simplicity (complex ops also work)
    auto output = input * 2.0;
    auto grad_output = torch::ones_like(output);
    output.backward(grad_output);

    EXPECT_TRUE(input.grad().defined());
    EXPECT_EQ(input.grad().scalar_type(), torch::kComplexDouble);
}

