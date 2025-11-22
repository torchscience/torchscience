#include <gtest/gtest.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

// Include the operator header to access CUDA kernel functions directly
#include "../../../../../src/torchscience/csrc/ops/example.h"

// Test fixture for CUDA tests
class CUDAExampleKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Skip tests if CUDA is not available
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available, skipping CUDA tests";
        }
        torch::manual_seed(42);
    }
};

TEST_F(CUDAExampleKernelTest, BasicOperation) {
    auto input = torch::randn({5, 5}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    double scalar = 2.0;

    auto result = science::ops::cuda::example_forward_kernel(input, scalar);

    // Verify result properties
    EXPECT_EQ(result.sizes(), input.sizes());
    EXPECT_EQ(result.dtype(), input.dtype());
    EXPECT_TRUE(result.device().is_cuda());

    // Verify mathematical correctness (compare on CPU)
    auto expected = input + scalar;
    EXPECT_TRUE(torch::allclose(result, expected));
}

TEST_F(CUDAExampleKernelTest, HandlesEmptyTensor) {
    auto input = torch::empty({0, 5}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    auto result = science::ops::cuda::example_forward_kernel(input, 2.0);

    EXPECT_EQ(result.sizes(), input.sizes());
    EXPECT_EQ(result.numel(), 0);
    EXPECT_TRUE(result.device().is_cuda());
}

TEST_F(CUDAExampleKernelTest, HandlesLargeTensors) {
    // Test with tensor larger than single block can handle
    auto input = torch::randn({10000, 1000}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    auto result = science::ops::cuda::example_forward_kernel(input, 3.14);

    EXPECT_EQ(result.sizes(), input.sizes());

    auto expected = input + 3.14;
    EXPECT_TRUE(torch::allclose(result, expected));
}

// Test different dtypes on CUDA
class CUDAExampleDtypeTest : public ::testing::TestWithParam<torch::ScalarType> {
protected:
    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
    }
};

TEST_P(CUDAExampleDtypeTest, PreservesDtype) {
    auto dtype = GetParam();
    auto input = torch::randn(
        {100, 100},
        torch::device(torch::kCUDA).dtype(dtype)
    );
    auto result = science::ops::cuda::example_forward_kernel(input, 1.5);

    EXPECT_EQ(result.dtype(), dtype);
    EXPECT_EQ(result.sizes(), input.sizes());
    EXPECT_TRUE(result.device().is_cuda());
}

INSTANTIATE_TEST_SUITE_P(
    CUDADtypes,
    CUDAExampleDtypeTest,
    ::testing::Values(
        torch::kFloat32,
        torch::kFloat64,
        torch::kFloat16,
        torch::kBFloat16,
        torch::kInt32,
        torch::kInt64
    )
);

// Test CPU-CUDA transfer correctness
TEST_F(CUDAExampleKernelTest, CPUCUDAConsistency) {
    // Create same input on CPU and CUDA
    auto input_cpu = torch::randn({100, 100}, torch::kFloat32);
    auto input_cuda = input_cpu.cuda();
    double scalar = 2.718;

    // Run on both devices
    auto result_cpu = science::ops::cpu::example_forward_kernel(input_cpu, scalar);
    auto result_cuda = science::ops::cuda::example_forward_kernel(input_cuda, scalar);

    // Results should match when moved to same device
    EXPECT_TRUE(torch::allclose(result_cpu, result_cuda.cpu()));
}

// Test backward pass on CUDA
TEST_F(CUDAExampleKernelTest, BackwardPassCorrectness) {
    auto input = torch::randn(
        {10, 10},
        torch::device(torch::kCUDA).dtype(torch::kFloat64)
    );
    auto grad_output = torch::ones({10, 10}, torch::device(torch::kCUDA).dtype(torch::kFloat64));

    // Test backward kernel directly
    auto grad_input = science::ops::cuda::example_backward_kernel(grad_output, input, 1.0);

    // Gradient should equal grad_output (derivative is 1)
    EXPECT_TRUE(grad_input.device().is_cuda());
    EXPECT_TRUE(torch::allclose(grad_input, grad_output));
}

// Test non-contiguous tensors on CUDA
TEST_F(CUDAExampleKernelTest, HandlesNonContiguousTensors) {
    auto input = torch::randn(
        {100, 100},
        torch::device(torch::kCUDA).dtype(torch::kFloat32)
    ).transpose(0, 1);

    EXPECT_FALSE(input.is_contiguous());

    auto result = science::ops::cuda::example_forward_kernel(input, 2.0);

    // Verify correctness
    auto expected = input + 2.0;
    EXPECT_TRUE(torch::allclose(result, expected));
}

// Multi-GPU test (if available)
TEST_F(CUDAExampleKernelTest, MultiGPUSupport) {
    if (torch::cuda::device_count() < 2) {
        GTEST_SKIP() << "Less than 2 GPUs available, skipping multi-GPU test";
    }

    // Test on GPU 0
    {
        c10::cuda::CUDAGuard guard(0);
        auto input = torch::randn({50, 50}, torch::device(torch::kCUDA, 0).dtype(torch::kFloat32));
        auto result = science::ops::cuda::example_forward_kernel(input, 1.0);

        EXPECT_EQ(result.device().index(), 0);
        EXPECT_TRUE(torch::allclose(result, input + 1.0));
    }

    // Test on GPU 1
    {
        c10::cuda::CUDAGuard guard(1);
        auto input = torch::randn({50, 50}, torch::device(torch::kCUDA, 1).dtype(torch::kFloat32));
        auto result = science::ops::cuda::example_forward_kernel(input, 2.0);

        EXPECT_EQ(result.device().index(), 1);
        EXPECT_TRUE(torch::allclose(result, input + 2.0));
    }
}

// Stress test: very large tensor
TEST_F(CUDAExampleKernelTest, VeryLargeTensorStressTest) {
    // Test with >1GB tensor (256M floats = 1GB)
    try {
        auto input = torch::randn(
            {16384, 16384},  // 256M elements
            torch::device(torch::kCUDA).dtype(torch::kFloat32)
        );
        auto result = science::ops::cuda::example_forward_kernel(input, 1.0);

        EXPECT_EQ(result.sizes(), input.sizes());
        EXPECT_TRUE(result.device().is_cuda());

        // Sample check (full check would be slow)
        auto sample = result.flatten()[torch::indexing::Slice(0, 1000)];
        auto expected_sample = (input.flatten()[torch::indexing::Slice(0, 1000)]) + 1.0;
        EXPECT_TRUE(torch::allclose(sample, expected_sample));
    } catch (const c10::Error& e) {
        // Skip if out of memory
        if (std::string(e.what()).find("out of memory") != std::string::npos) {
            GTEST_SKIP() << "Out of GPU memory for large tensor test";
        } else {
            throw;
        }
    }
}

// Test numerical precision on CUDA
TEST_F(CUDAExampleKernelTest, NumericalPrecisionFloat64) {
    auto input = torch::randn(
        {100, 100},
        torch::device(torch::kCUDA).dtype(torch::kFloat64)
    );
    double scalar = 1.23456789012345;

    auto result = science::ops::cuda::example_forward_kernel(input, scalar);
    auto expected = input + scalar;

    // Very tight tolerance for float64
    EXPECT_TRUE(torch::allclose(result, expected, /*rtol=*/1e-12, /*atol=*/1e-14));
}

// Test CUDA helper functions
TEST(CUDAHelperTest, GetNumThreads) {
    EXPECT_EQ(science::ops::cuda::get_num_threads(), 256);
}

TEST(CUDAHelperTest, GetNumBlocks) {
    EXPECT_EQ(science::ops::cuda::get_num_blocks(100, 256), 1);
    EXPECT_EQ(science::ops::cuda::get_num_blocks(1000, 256), 4);
    EXPECT_EQ(science::ops::cuda::get_num_blocks(100000, 256), 391);

    // Test max blocks limit
    int64_t huge_numel = 100000000LL;
    EXPECT_LE(science::ops::cuda::get_num_blocks(huge_numel, 256), 65535);
}
