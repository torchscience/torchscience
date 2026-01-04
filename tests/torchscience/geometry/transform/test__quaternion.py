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

    def test_non_commutative(self):
        """Quaternion multiplication is non-commutative."""
        import math

        # Two different rotations
        q1 = quaternion(
            torch.tensor(
                [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]
            )
        )  # 90 deg around x
        q2 = quaternion(
            torch.tensor(
                [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]
            )
        )  # 90 deg around y
        result1 = quaternion_multiply(q1, q2)
        result2 = quaternion_multiply(q2, q1)
        assert not torch.allclose(result1.wxyz, result2.wxyz, atol=1e-5)


class TestQuaternionMultiplyShape:
    """Tests for quaternion_multiply shape handling."""

    def test_single_quaternion(self):
        """Single quaternion (4,) input."""
        q1 = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        q2 = quaternion(torch.tensor([0.7071, 0.0, 0.0, 0.7071]))
        result = quaternion_multiply(q1, q2)
        assert result.wxyz.shape == (4,)

    def test_batch(self):
        """Batch of quaternions (B, 4)."""
        q1 = quaternion(torch.randn(10, 4))
        q2 = quaternion(torch.randn(10, 4))
        result = quaternion_multiply(q1, q2)
        assert result.wxyz.shape == (10, 4)

    def test_image_shape(self):
        """Image-like shape (H, W, 4)."""
        q1 = quaternion(torch.randn(64, 64, 4))
        q2 = quaternion(torch.randn(64, 64, 4))
        result = quaternion_multiply(q1, q2)
        assert result.wxyz.shape == (64, 64, 4)

    def test_invalid_last_dim(self):
        """Raise error if last dimension is not 4."""
        with pytest.raises(ValueError, match="last dimension 4"):
            quaternion(torch.randn(10, 3))


class TestQuaternionMultiplyGradients:
    """Tests for quaternion_multiply gradient computation."""

    def test_gradcheck_q1(self):
        """Gradient check w.r.t. q1."""
        q1 = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
        q2 = torch.randn(5, 4, dtype=torch.float64)
        assert gradcheck(
            lambda a: quaternion_multiply(
                Quaternion(wxyz=a), Quaternion(wxyz=q2)
            ).wxyz,
            (q1,),
            eps=1e-6,
            atol=1e-4,
        )

    def test_gradcheck_q2(self):
        """Gradient check w.r.t. q2."""
        q1 = torch.randn(5, 4, dtype=torch.float64)
        q2 = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda b: quaternion_multiply(
                Quaternion(wxyz=q1), Quaternion(wxyz=b)
            ).wxyz,
            (q2,),
            eps=1e-6,
            atol=1e-4,
        )

    def test_gradcheck_both(self):
        """Gradient check w.r.t. both inputs."""
        q1 = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
        q2 = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda a, b: quaternion_multiply(
                Quaternion(wxyz=a), Quaternion(wxyz=b)
            ).wxyz,
            (q1, q2),
            eps=1e-6,
            atol=1e-4,
        )

    def test_gradcheck_broadcast(self):
        """Gradient check with broadcasting."""
        q1 = torch.randn(3, 1, 4, dtype=torch.float64, requires_grad=True)
        q2 = torch.randn(1, 2, 4, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda a, b: quaternion_multiply(
                Quaternion(wxyz=a), Quaternion(wxyz=b)
            ).wxyz,
            (q1, q2),
            eps=1e-6,
            atol=1e-4,
        )


class TestQuaternionMultiplyDtypes:
    """Tests for quaternion_multiply with different data types."""

    def test_float32(self):
        """Works with float32."""
        q1 = quaternion(torch.randn(10, 4, dtype=torch.float32))
        q2 = quaternion(torch.randn(10, 4, dtype=torch.float32))
        result = quaternion_multiply(q1, q2)
        assert result.wxyz.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        q1 = quaternion(torch.randn(10, 4, dtype=torch.float64))
        q2 = quaternion(torch.randn(10, 4, dtype=torch.float64))
        result = quaternion_multiply(q1, q2)
        assert result.wxyz.dtype == torch.float64

    def test_bfloat16(self):
        """Works with bfloat16."""
        q1 = quaternion(torch.randn(10, 4, dtype=torch.bfloat16))
        q2 = quaternion(torch.randn(10, 4, dtype=torch.bfloat16))
        result = quaternion_multiply(q1, q2)
        assert result.wxyz.dtype == torch.bfloat16

    def test_float16(self):
        """Works with float16."""
        q1 = quaternion(torch.randn(10, 4, dtype=torch.float16))
        q2 = quaternion(torch.randn(10, 4, dtype=torch.float16))
        result = quaternion_multiply(q1, q2)
        assert result.wxyz.dtype == torch.float16

    def test_dtype_mismatch_error(self):
        """Raises error when dtypes don't match."""
        q1 = quaternion(torch.randn(10, 4, dtype=torch.float32))
        q2 = quaternion(torch.randn(10, 4, dtype=torch.float64))
        with pytest.raises(RuntimeError, match="same dtype"):
            quaternion_multiply(q1, q2)
