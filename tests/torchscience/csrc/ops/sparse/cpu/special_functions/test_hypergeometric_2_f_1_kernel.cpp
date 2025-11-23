#include "ops/special_functions.h"

#include <ATen/ATen.h>
#include <torch/torch.h>

#include <gtest/gtest.h>

#include <cmath>

// Test fixture for hypergeometric_2_f_1 Sparse CPU kernel tests
class Hypergeometric2F1SparseCPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        // No special setup needed for sparse CPU tests
    }

    // Helper to create COO sparse tensor from simple 1D data
    at::Tensor create_sparse_tensor_1d(const std::vector<int64_t>& indices_data,
                                        const std::vector<double>& values_data,
                                        int64_t size) {
        // Create indices tensor (2D: [1, nnz])
        auto indices = torch::zeros({1, static_cast<int64_t>(indices_data.size())}, torch::kInt64);
        for (size_t i = 0; i < indices_data.size(); ++i) {
            indices[0][i] = indices_data[i];
        }

        // Create values tensor
        auto values = torch::zeros({static_cast<int64_t>(values_data.size())}, torch::kFloat64);
        for (size_t i = 0; i < values_data.size(); ++i) {
            values[i] = values_data[i];
        }

        return torch::sparse_coo_tensor(indices, values, {size}, torch::kFloat64);
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

TEST_F(Hypergeometric2F1SparseCPUTest, BasicFunctionality) {
    // Test basic functionality with simple sparse tensors
    auto a = create_sparse_tensor_1d({0, 1, 2}, {1.0, 2.0, 3.0}, 5);
    auto b = create_sparse_tensor_1d({0, 1, 2}, {2.0, 3.0, 4.0}, 5);
    auto c = create_sparse_tensor_1d({0, 1, 2}, {3.0, 4.0, 5.0}, 5);
    auto z = create_sparse_tensor_1d({0, 1, 2}, {0.5, 0.3, 0.2}, 5);

    auto result = science::ops::sparse::cpu::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_TRUE(result.is_sparse());
    EXPECT_EQ(result.scalar_type(), torch::kFloat64);
    EXPECT_TRUE(result.device().is_cpu());
}

TEST_F(Hypergeometric2F1SparseCPUTest, SparseStructurePreservation) {
    // Test that sparse structure (indices) is preserved
    std::vector<int64_t> indices_data = {0, 2, 4};
    auto a = create_sparse_tensor_1d(indices_data, {1.0, 2.0, 3.0}, 10);
    auto b = create_sparse_tensor_1d(indices_data, {2.0, 3.0, 4.0}, 10);
    auto c = create_sparse_tensor_1d(indices_data, {3.0, 4.0, 5.0}, 10);
    auto z = create_sparse_tensor_1d(indices_data, {0.3, 0.4, 0.5}, 10);

    auto result = science::ops::sparse::cpu::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    // Check that indices are preserved
    auto result_indices = result._indices();
    auto z_indices = z._indices();
    EXPECT_TRUE(torch::equal(result_indices, z_indices));

    // Check nnz (number of non-zero elements)
    EXPECT_EQ(result._nnz(), z._nnz());
}

TEST_F(Hypergeometric2F1SparseCPUTest, DenseSparseConsistency) {
    // Create same data as dense and sparse
    std::vector<double> values_a = {1.0, 2.0, 3.0};
    std::vector<double> values_b = {2.0, 3.0, 4.0};
    std::vector<double> values_c = {3.0, 4.0, 5.0};
    std::vector<double> values_z = {0.3, 0.4, 0.5};

    // Dense version
    auto a_dense = torch::tensor(values_a, torch::kFloat64);
    auto b_dense = torch::tensor(values_b, torch::kFloat64);
    auto c_dense = torch::tensor(values_c, torch::kFloat64);
    auto z_dense = torch::tensor(values_z, torch::kFloat64);

    auto result_dense = science::ops::cpu::hypergeometric_2_f_1_forward_kernel(
        a_dense, b_dense, c_dense, z_dense);

    // Sparse version
    auto a_sparse = create_sparse_tensor_1d({0, 1, 2}, values_a, 3);
    auto b_sparse = create_sparse_tensor_1d({0, 1, 2}, values_b, 3);
    auto c_sparse = create_sparse_tensor_1d({0, 1, 2}, values_c, 3);
    auto z_sparse = create_sparse_tensor_1d({0, 1, 2}, values_z, 3);

    auto result_sparse = science::ops::sparse::cpu::hypergeometric_2_f_1_forward_kernel(
        a_sparse, b_sparse, c_sparse, z_sparse);

    // Compare values
    EXPECT_TRUE(torch::allclose(result_sparse._values(), result_dense, /*rtol=*/1e-10, /*atol=*/1e-12));
}

TEST_F(Hypergeometric2F1SparseCPUTest, EmptySparseTensor) {
    // Test with sparse tensors that have no non-zero elements
    auto a = create_sparse_tensor_1d({}, {}, 10);
    auto b = create_sparse_tensor_1d({}, {}, 10);
    auto c = create_sparse_tensor_1d({}, {}, 10);
    auto z = create_sparse_tensor_1d({}, {}, 10);

    auto result = science::ops::sparse::cpu::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_TRUE(result.is_sparse());
    EXPECT_EQ(result._nnz(), 0);
}

TEST_F(Hypergeometric2F1SparseCPUTest, Float32Dtype) {
    // Create indices tensor
    auto indices = torch::zeros({1, 3}, torch::kInt64);
    indices[0][0] = 0;
    indices[0][1] = 1;
    indices[0][2] = 2;

    // Create values tensor (float32)
    auto values = torch::tensor({1.0f, 2.0f, 3.0f}, torch::kFloat32);

    auto a = torch::sparse_coo_tensor(indices, values, {5}, torch::kFloat32);
    auto b = torch::sparse_coo_tensor(indices, torch::tensor({2.0f, 3.0f, 4.0f}, torch::kFloat32), {5}, torch::kFloat32);
    auto c = torch::sparse_coo_tensor(indices, torch::tensor({3.0f, 4.0f, 5.0f}, torch::kFloat32), {5}, torch::kFloat32);
    auto z = torch::sparse_coo_tensor(indices, torch::tensor({0.3f, 0.4f, 0.5f}, torch::kFloat32), {5}, torch::kFloat32);

    auto result = science::ops::sparse::cpu::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

    EXPECT_TRUE(result.is_sparse());
    EXPECT_EQ(result.scalar_type(), torch::kFloat32);
}

TEST_F(Hypergeometric2F1SparseCPUTest, BackwardShapeCorrectness) {
    // Test backward pass shape correctness
    auto a = create_sparse_tensor_1d({0, 1, 2}, {1.0, 2.0, 3.0}, 5);
    auto b = create_sparse_tensor_1d({0, 1, 2}, {2.0, 3.0, 4.0}, 5);
    auto c = create_sparse_tensor_1d({0, 1, 2}, {3.0, 4.0, 5.0}, 5);
    auto z = create_sparse_tensor_1d({0, 1, 2}, {0.3, 0.4, 0.5}, 5);

    auto result = science::ops::sparse::cpu::hypergeometric_2_f_1_forward_kernel(a, b, c, z);
    auto grad_out = create_sparse_tensor_1d({0, 1, 2}, {1.0, 1.0, 1.0}, 5);

    auto grads = science::ops::sparse::cpu::hypergeometric_2_f_1_backward_kernel(
        grad_out, a, b, c, z, result);

    // Check that all gradients have correct shapes
    EXPECT_TRUE(std::get<0>(grads).is_sparse());
    EXPECT_TRUE(std::get<1>(grads).is_sparse());
    EXPECT_TRUE(std::get<2>(grads).is_sparse());
    EXPECT_TRUE(std::get<3>(grads).is_sparse());

    EXPECT_EQ(std::get<0>(grads).sizes(), a.sizes());
    EXPECT_EQ(std::get<1>(grads).sizes(), b.sizes());
    EXPECT_EQ(std::get<2>(grads).sizes(), c.sizes());
    EXPECT_EQ(std::get<3>(grads).sizes(), z.sizes());
}

TEST_F(Hypergeometric2F1SparseCPUTest, NumericalStability) {
    // Test with various input ranges
    std::vector<std::vector<double>> test_cases = {
        {0.5, 1.5, 2.0, 0.3},  // Standard values
        {0.1, 0.2, 1.0, 0.5},  // Small a, b
        {5.0, 3.0, 8.0, 0.2},  // Larger values
    };

    for (const auto& test_case : test_cases) {
        auto a = create_sparse_tensor_1d({0}, {test_case[0]}, 3);
        auto b = create_sparse_tensor_1d({0}, {test_case[1]}, 3);
        auto c = create_sparse_tensor_1d({0}, {test_case[2]}, 3);
        auto z = create_sparse_tensor_1d({0}, {test_case[3]}, 3);

        auto result = science::ops::sparse::cpu::hypergeometric_2_f_1_forward_kernel(a, b, c, z);

        // Result should be finite
        EXPECT_TRUE(torch::isfinite(result._values()).all().item<bool>())
            << "Result contains non-finite values for test case: "
            << test_case[0] << ", " << test_case[1] << ", " << test_case[2] << ", " << test_case[3];
    }
}

