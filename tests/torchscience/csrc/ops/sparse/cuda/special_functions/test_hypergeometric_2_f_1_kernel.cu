#include <gtest/gtest.h>

#include "ops/special_functions.h"

#include <ATen/ATen.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>

#include <cmath>

// Test fixture for hypergeometric_2_f_1 Sparse CUDA kernel tests
class Hypergeometric2F1SparseCUDATest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if CUDA is available
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available, skipping Sparse CUDA tests";
        }
    }

    // Helper to create COO sparse tensor on CUDA
    at::Tensor create_sparse_tensor_cuda(const std::vector<std::vector<int64_t>>& indices_vec,
                                          const std::vector<double>& values_vec,
                                          const std::vector<int64_t>& shape) {
        auto indices = torch::tensor(indices_vec, torch::kInt64);
        auto values = torch::tensor(values_vec, torch::kFloat64);
        return torch::sparse_coo_tensor(indices, values, shape, torch::dtype(torch::kFloat64).device(torch::kCUDA));
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

TEST_F(Hypergeometric2F1SparseCUDATest, BasicFunctionality) {
    // Test basic functionality with simple sparse tensors on CUDA
    auto a = create_sparse_tensor_cuda({{0, 1, 2}}, {1.0, 2.0, 3.0}, {5});
    auto b = create_sparse_tensor_cuda({{0, 1, 2}}, {2.0, 3.0, 4.0}, {5});
    auto c = create_sparse_tensor_cuda({{0, 1, 2}}, {3.0, 4.0, 5.0}, {5});
    auto z = create_sparse_tensor_cuda({{0, 1, 2}}, {0.5, 0.3, 0.2}, {5});

    auto result = science::ops::sparse_cuda::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_TRUE(result.is_sparse());
    EXPECT_EQ(result.scalar_type(), torch::kFloat64);
    EXPECT_TRUE(result.device().is_cuda());
}

TEST_F(Hypergeometric2F1SparseCUDATest, SparseStructurePreservation) {
    // Test that sparse structure (indices) is preserved
    auto a = create_sparse_tensor_cuda({{0, 2, 4}}, {1.0, 2.0, 3.0}, {10});
    auto b = create_sparse_tensor_cuda({{0, 2, 4}}, {2.0, 3.0, 4.0}, {10});
    auto c = create_sparse_tensor_cuda({{0, 2, 4}}, {3.0, 4.0, 5.0}, {10});
    auto z = create_sparse_tensor_cuda({{0, 2, 4}}, {0.1, 0.2, 0.3}, {10});

    auto result = science::ops::sparse_cuda::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    // Check that indices are preserved
    auto result_indices = result._indices();
    auto expected_indices = a._indices();

    EXPECT_TRUE(torch::equal(result_indices, expected_indices))
        << "Sparse indices not preserved";
}

TEST_F(Hypergeometric2F1SparseCUDATest, ValuesDimsPreserved) {
    // Test that values dimensions are preserved
    auto a = create_sparse_tensor_cuda({{0, 1, 2, 3}}, {1.0, 2.0, 3.0, 4.0}, {5});
    auto b = create_sparse_tensor_cuda({{0, 1, 2, 3}}, {2.0, 3.0, 4.0, 5.0}, {5});
    auto c = create_sparse_tensor_cuda({{0, 1, 2, 3}}, {3.0, 4.0, 5.0, 6.0}, {5});
    auto z = create_sparse_tensor_cuda({{0, 1, 2, 3}}, {0.1, 0.2, 0.3, 0.4}, {5});

    auto result = science::ops::sparse_cuda::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    auto result_values = result._values();
    auto input_values = a._values();

    EXPECT_EQ(result_values.sizes(), input_values.sizes())
        << "Values dimensions not preserved";
}

TEST_F(Hypergeometric2F1SparseCUDATest, EmptySparseTensor) {
    // Test with empty sparse tensors (no non-zero elements)
    auto a = create_sparse_tensor_cuda({{}, {}}, {}, {10, 10});
    auto b = create_sparse_tensor_cuda({{}, {}}, {}, {10, 10});
    auto c = create_sparse_tensor_cuda({{}, {}}, {}, {10, 10});
    auto z = create_sparse_tensor_cuda({{}, {}}, {}, {10, 10});

    auto result = science::ops::sparse_cuda::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_TRUE(result.is_sparse());
    EXPECT_EQ(result._values().numel(), 0);
}

TEST_F(Hypergeometric2F1SparseCUDATest, TwoDimensionalSparseTensor) {
    // Test with 2D sparse tensors
    auto a = create_sparse_tensor_cuda({{0, 1, 2}, {0, 1, 2}}, {1.0, 2.0, 3.0}, {5, 5});
    auto b = create_sparse_tensor_cuda({{0, 1, 2}, {0, 1, 2}}, {2.0, 3.0, 4.0}, {5, 5});
    auto c = create_sparse_tensor_cuda({{0, 1, 2}, {0, 1, 2}}, {3.0, 4.0, 5.0}, {5, 5});
    auto z = create_sparse_tensor_cuda({{0, 1, 2}, {0, 1, 2}}, {0.1, 0.2, 0.3}, {5, 5});

    auto result = science::ops::sparse_cuda::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_TRUE(result.is_sparse());
    EXPECT_EQ(result.sizes(), torch::IntArrayRef({5, 5}));
}

TEST_F(Hypergeometric2F1SparseCUDATest, BackwardShapeCorrectness) {
    // Test backward pass returns correctly shaped gradients for sparse tensors
    auto grad_out = create_sparse_tensor_cuda({{0, 1, 2}}, {1.0, 2.0, 3.0}, {5});
    auto a = create_sparse_tensor_cuda({{0, 1, 2}}, {1.0, 2.0, 3.0}, {5});
    auto b = create_sparse_tensor_cuda({{0, 1, 2}}, {2.0, 3.0, 4.0}, {5});
    auto c = create_sparse_tensor_cuda({{0, 1, 2}}, {3.0, 4.0, 5.0}, {5});
    auto z = create_sparse_tensor_cuda({{0, 1, 2}}, {0.1, 0.2, 0.3}, {5});
    auto result = create_sparse_tensor_cuda({{0, 1, 2}}, {1.5, 2.5, 3.5}, {5});  // Placeholder

    auto grads = science::ops::sparse_cuda::hypergeometric_2_f_1_backward_kernel(
        grad_out, a, b, c, z, result);

    auto grad_a = std::get<0>(grads);
    auto grad_b = std::get<1>(grads);
    auto grad_c = std::get<2>(grads);
    auto grad_z = std::get<3>(grads);

    EXPECT_TRUE(grad_a.is_sparse());
    EXPECT_TRUE(grad_b.is_sparse());
    EXPECT_TRUE(grad_c.is_sparse());
    EXPECT_TRUE(grad_z.is_sparse());

    EXPECT_EQ(grad_a.sizes(), a.sizes());
    EXPECT_EQ(grad_b.sizes(), b.sizes());
    EXPECT_EQ(grad_c.sizes(), c.sizes());
    EXPECT_EQ(grad_z.sizes(), z.sizes());
}

TEST_F(Hypergeometric2F1SparseCUDATest, CPUCUDAConsistency) {
    // Test that sparse CPU and sparse CUDA implementations give consistent results
    // Create sparse tensors on CPU
    auto indices = torch::tensor({{0, 2, 4}}, torch::kInt64);
    auto values = torch::tensor({1.0, 2.0, 3.0}, torch::kFloat64);
    auto a_cpu = torch::sparse_coo_tensor(indices, values, {5}, torch::kFloat64);

    values = torch::tensor({2.0, 3.0, 4.0}, torch::kFloat64);
    auto b_cpu = torch::sparse_coo_tensor(indices, values, {5}, torch::kFloat64);

    values = torch::tensor({3.0, 4.0, 5.0}, torch::kFloat64);
    auto c_cpu = torch::sparse_coo_tensor(indices, values, {5}, torch::kFloat64);

    values = torch::tensor({0.1, 0.2, 0.3}, torch::kFloat64);
    auto z_cpu = torch::sparse_coo_tensor(indices, values, {5}, torch::kFloat64);

    // Move to CUDA
    auto a_cuda = a_cpu.cuda();
    auto b_cuda = b_cpu.cuda();
    auto c_cuda = c_cpu.cuda();
    auto z_cuda = z_cpu.cuda();

    auto result_cpu = science::ops::sparse::cuda::hypergeometric_2_f_1_forward_kernel(a_cpu, b_cpu, c_cpu, z_cpu);
    auto result_cuda = science::ops::sparse_cuda::hypergeometric_2_f_1_forward_kernel(a_cuda, b_cuda, c_cuda, z_cuda);

    // Move CUDA result back to CPU for comparison
    auto result_cuda_cpu = result_cuda.cpu();

    // Compare values
    auto result_cpu_values = result_cpu._values();
    auto result_cuda_values = result_cuda_cpu._values();

    EXPECT_TRUE(torch::allclose(result_cpu_values, result_cuda_values, /*rtol=*/1e-10, /*atol=*/1e-12))
        << "Sparse CPU and CUDA results differ";
}

TEST_F(Hypergeometric2F1SparseCUDATest, LargeSparseStressTest) {
    // Test with large sparse tensors to stress GPU parallelization
    std::vector<int64_t> indices_vec;
    std::vector<double> values_vec;
    for (int i = 0; i < 10000; i += 10) {
        indices_vec.push_back(i);
        values_vec.push_back(static_cast<double>(i) / 100000.0);
    }

    auto a = create_sparse_tensor_cuda({indices_vec}, values_vec, {100000});
    auto b = create_sparse_tensor_cuda({indices_vec}, values_vec, {100000});
    auto c = create_sparse_tensor_cuda({indices_vec}, values_vec, {100000});
    auto z = create_sparse_tensor_cuda({indices_vec}, values_vec, {100000});

    auto result = science::ops::sparse_cuda::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_TRUE(result.is_sparse());
    EXPECT_TRUE(torch::isfinite(result._values()).all().item<bool>())
        << "Large sparse tensor computation produced non-finite values";
}

TEST_F(Hypergeometric2F1SparseCUDATest, NumericalStability) {
    // Test numerical stability with various sparse input values
    auto test_values = std::vector<double>{0.1, 0.5, 0.9, -0.5, -1.0};

    for (size_t i = 0; i < test_values.size(); ++i) {
        auto a = create_sparse_tensor_cuda({{static_cast<int64_t>(i)}}, {1.0}, {10});
        auto b = create_sparse_tensor_cuda({{static_cast<int64_t>(i)}}, {2.0}, {10});
        auto c = create_sparse_tensor_cuda({{static_cast<int64_t>(i)}}, {3.0}, {10});
        auto z = create_sparse_tensor_cuda({{static_cast<int64_t>(i)}}, {test_values[i]}, {10});

        auto result = science::ops::sparse_cuda::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

        // Result values should be finite
        EXPECT_TRUE(torch::isfinite(result._values()).all().item<bool>())
            << "Result not finite for z=" << test_values[i];
    }
}
