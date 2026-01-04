"""Tests for Quaternion tensorclass."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.geometry.transform import (
    Quaternion,
    quaternion,
    quaternion_multiply,
)


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


class TestQuaternionMultiply:
    """Tests for quaternion_multiply."""

    def test_identity_left(self):
        """Multiplying by identity on left returns original."""
        identity = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        q = quaternion(torch.tensor([0.7071, 0.7071, 0.0, 0.0]))
        result = quaternion_multiply(identity, q)
        assert torch.allclose(result.wxyz, q.wxyz, atol=1e-5)

    def test_identity_right(self):
        """Multiplying by identity on right returns original."""
        identity = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        q = quaternion(torch.tensor([0.7071, 0.7071, 0.0, 0.0]))
        result = quaternion_multiply(q, identity)
        assert torch.allclose(result.wxyz, q.wxyz, atol=1e-5)

    def test_90_deg_rotations(self):
        """Two 90-degree rotations around z = 180-degree rotation."""
        # 90 degrees around z: [cos(45), 0, 0, sin(45)]
        q90z = quaternion(
            torch.tensor([0.7071067811865476, 0.0, 0.0, 0.7071067811865476])
        )
        result = quaternion_multiply(q90z, q90z)
        # 180 degrees around z: [0, 0, 0, 1]
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0])
        assert torch.allclose(result.wxyz, expected, atol=1e-5)

    def test_inverse_gives_identity(self):
        """q * q^(-1) = identity (conjugate for unit quaternion)."""
        q = quaternion(torch.tensor([0.5, 0.5, 0.5, 0.5]))
        q_conj = quaternion(torch.tensor([0.5, -0.5, -0.5, -0.5]))
        result = quaternion_multiply(q, q_conj)
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
        assert torch.allclose(result.wxyz, expected, atol=1e-5)

    def test_batch(self):
        """Batched multiplication."""
        q1 = quaternion(torch.randn(10, 4))
        q2 = quaternion(torch.randn(10, 4))
        result = quaternion_multiply(q1, q2)
        assert result.wxyz.shape == (10, 4)

    def test_broadcast(self):
        """Broadcasting batch dimensions."""
        q1 = quaternion(torch.randn(5, 1, 4))
        q2 = quaternion(torch.randn(1, 3, 4))
        result = quaternion_multiply(q1, q2)
        assert result.wxyz.shape == (5, 3, 4)

    def test_gradcheck(self):
        """Gradient check."""
        q1 = quaternion(
            torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
        )
        q2 = quaternion(
            torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
        )
        assert gradcheck(
            lambda a, b: quaternion_multiply(
                Quaternion(wxyz=a), Quaternion(wxyz=b)
            ).wxyz,
            (q1.wxyz, q2.wxyz),
            eps=1e-6,
            atol=1e-4,
        )

    @pytest.mark.skip(
        reason="Second-order gradients not yet implemented for quaternion_multiply"
    )
    def test_gradgradcheck(self):
        """Second-order gradient check."""
        from torch.autograd import gradgradcheck

        q1 = quaternion(
            torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        )
        q2 = quaternion(
            torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        )
        assert gradgradcheck(
            lambda a, b: quaternion_multiply(
                Quaternion(wxyz=a), Quaternion(wxyz=b)
            ).wxyz,
            (q1.wxyz, q2.wxyz),
            eps=1e-6,
            atol=1e-4,
        )
