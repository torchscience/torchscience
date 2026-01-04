"""Tests for Quaternion tensorclass."""

import pytest
import torch

from torchscience.geometry.transform import Quaternion, quaternion


class TestQuaternionConstruction:
    """Tests for Quaternion construction."""

    def test_from_tensor(self):
        """Create Quaternion from tensor."""
        wxyz = torch.tensor([1.0, 0.0, 0.0, 0.0])
        q = Quaternion(wxyz=wxyz)
        assert q.wxyz.shape == (4,)
        assert torch.allclose(q.wxyz, wxyz)

    def test_batch(self):
        """Batch of quaternions."""
        wxyz = torch.randn(10, 4)
        q = Quaternion(wxyz=wxyz)
        assert q.wxyz.shape == (10, 4)

    def test_factory_function(self):
        """Create via quaternion() factory."""
        wxyz = torch.tensor([1.0, 0.0, 0.0, 0.0])
        q = quaternion(wxyz)
        assert isinstance(q, Quaternion)
        assert torch.allclose(q.wxyz, wxyz)

    def test_invalid_shape(self):
        """Raise error for wrong last dimension."""
        with pytest.raises(ValueError, match="last dimension 4"):
            quaternion(torch.randn(3))
