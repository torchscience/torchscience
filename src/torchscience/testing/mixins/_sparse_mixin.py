from typing import TYPE_CHECKING

import pytest
import torch
import torch.testing

if TYPE_CHECKING:
    from torchscience.testing.descriptors import OperatorDescriptor


class SparseMixin:
    """Mixin providing sparse tensor tests."""

    descriptor: "OperatorDescriptor"

    def test_sparse_coo_basic(self):
        """Test basic sparse COO tensor support."""
        if "test_sparse_coo_basic" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if not self.descriptor.supports_sparse_coo:
            pytest.skip("Operator does not support sparse COO")

        if self.descriptor.arity != 1:
            pytest.skip("Sparse test only for unary operators")

        # Create sparse COO tensor
        indices = torch.tensor([[0, 1, 3], [1, 2, 0]])
        values = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float64)
        sparse = torch.sparse_coo_tensor(indices, values, (4, 3))

        result = self.descriptor.func(sparse)

        # Verify sparsity is preserved
        assert result.is_sparse
        assert result.shape == sparse.shape

        # Compare with dense computation
        dense_result = self.descriptor.func(sparse.to_dense())
        rtol, atol = self.descriptor.tolerances.get_tolerances(torch.float64)
        torch.testing.assert_close(
            result.to_dense(), dense_result, rtol=rtol, atol=atol
        )

    def test_sparse_csr_basic(self):
        """Test basic sparse CSR tensor support."""
        if "test_sparse_csr_basic" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if not self.descriptor.supports_sparse_csr:
            pytest.skip("Operator does not support sparse CSR")

        if self.descriptor.arity != 1:
            pytest.skip("Sparse test only for unary operators")

        # Create sparse CSR tensor
        crow_indices = torch.tensor([0, 1, 2, 3])
        col_indices = torch.tensor([1, 2, 0])
        values = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float64)
        sparse = torch.sparse_csr_tensor(
            crow_indices, col_indices, values, (3, 3)
        )

        result = self.descriptor.func(sparse)

        # Verify sparsity is preserved
        assert result.layout == torch.sparse_csr
        assert result.shape == sparse.shape
