"""Tests for Quaternion tensorclass."""

import math

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.geometry.transform import (
    Quaternion,
    quaternion,
    quaternion_apply,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_normalize,
    quaternion_to_matrix,
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


class TestQuaternionInverse:
    """Tests for quaternion_inverse."""

    def test_inverse_of_identity(self):
        """Inverse of identity is identity."""
        identity = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        result = quaternion_inverse(identity)
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
        assert torch.allclose(result.wxyz, expected, atol=1e-5)

    def test_multiply_by_inverse_gives_identity(self):
        """q * q^(-1) = identity for unit quaternion."""
        q = quaternion(torch.tensor([0.5, 0.5, 0.5, 0.5]))
        q_inv = quaternion_inverse(q)
        result = quaternion_multiply(q, q_inv)
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
        assert torch.allclose(result.wxyz, expected, atol=1e-5)

    def test_inverse_multiply_gives_identity(self):
        """q^(-1) * q = identity for unit quaternion."""
        q = quaternion(torch.tensor([0.5, 0.5, 0.5, 0.5]))
        q_inv = quaternion_inverse(q)
        result = quaternion_multiply(q_inv, q)
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
        assert torch.allclose(result.wxyz, expected, atol=1e-5)

    def test_double_inverse(self):
        """(q^(-1))^(-1) = q."""
        q = quaternion(torch.tensor([0.5, 0.5, 0.5, 0.5]))
        q_inv_inv = quaternion_inverse(quaternion_inverse(q))
        assert torch.allclose(q_inv_inv.wxyz, q.wxyz, atol=1e-5)

    def test_batch(self):
        """Batched inverse."""
        q = quaternion(torch.randn(10, 4))
        result = quaternion_inverse(q)
        assert result.wxyz.shape == (10, 4)

    def test_conjugate_values(self):
        """Inverse equals conjugate: [w, -x, -y, -z]."""
        q = quaternion(torch.tensor([0.7071, 0.7071, 0.0, 0.0]))
        result = quaternion_inverse(q)
        expected = torch.tensor([0.7071, -0.7071, 0.0, 0.0])
        assert torch.allclose(result.wxyz, expected, atol=1e-4)

    def test_gradcheck(self):
        """Gradient check."""
        q = quaternion(
            torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
        )
        assert gradcheck(
            lambda a: quaternion_inverse(Quaternion(wxyz=a)).wxyz,
            (q.wxyz,),
            eps=1e-6,
            atol=1e-4,
        )


class TestQuaternionInverseShape:
    """Tests for quaternion_inverse shape handling."""

    def test_single_quaternion(self):
        """Single quaternion (4,) input."""
        q = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        result = quaternion_inverse(q)
        assert result.wxyz.shape == (4,)

    def test_batch(self):
        """Batch of quaternions (B, 4)."""
        q = quaternion(torch.randn(10, 4))
        result = quaternion_inverse(q)
        assert result.wxyz.shape == (10, 4)

    def test_image_shape(self):
        """Image-like shape (H, W, 4)."""
        q = quaternion(torch.randn(64, 64, 4))
        result = quaternion_inverse(q)
        assert result.wxyz.shape == (64, 64, 4)

    def test_multi_batch_shape(self):
        """Multi-batch shape (B, C, H, W, 4)."""
        q = quaternion(torch.randn(2, 3, 16, 16, 4))
        result = quaternion_inverse(q)
        assert result.wxyz.shape == (2, 3, 16, 16, 4)

    def test_invalid_last_dim(self):
        """Raise error if last dimension is not 4."""
        with pytest.raises(ValueError, match="last dimension 4"):
            quaternion_inverse(quaternion(torch.randn(10, 3)))


class TestQuaternionInverseGradients:
    """Tests for quaternion_inverse gradient computation."""

    def test_gradcheck(self):
        """Gradient check w.r.t. q."""
        q = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda a: quaternion_inverse(Quaternion(wxyz=a)).wxyz,
            (q,),
            eps=1e-6,
            atol=1e-4,
        )

    def test_gradcheck_batch(self):
        """Gradient check with batch."""
        q = torch.randn(3, 5, 4, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda a: quaternion_inverse(Quaternion(wxyz=a)).wxyz,
            (q,),
            eps=1e-6,
            atol=1e-4,
        )


class TestQuaternionInverseDtypes:
    """Tests for quaternion_inverse with different data types."""

    def test_float32(self):
        """Works with float32."""
        q = quaternion(torch.randn(10, 4, dtype=torch.float32))
        result = quaternion_inverse(q)
        assert result.wxyz.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        q = quaternion(torch.randn(10, 4, dtype=torch.float64))
        result = quaternion_inverse(q)
        assert result.wxyz.dtype == torch.float64

    def test_bfloat16(self):
        """Works with bfloat16."""
        q = quaternion(torch.randn(10, 4, dtype=torch.bfloat16))
        result = quaternion_inverse(q)
        assert result.wxyz.dtype == torch.bfloat16

    def test_float16(self):
        """Works with float16."""
        q = quaternion(torch.randn(10, 4, dtype=torch.float16))
        result = quaternion_inverse(q)
        assert result.wxyz.dtype == torch.float16


class TestQuaternionNormalize:
    """Tests for quaternion_normalize."""

    def test_normalize_identity(self):
        """Normalizing identity quaternion returns identity."""
        identity = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        result = quaternion_normalize(identity)
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
        assert torch.allclose(result.wxyz, expected, atol=1e-5)

    def test_normalize_scaled_quaternion(self):
        """Normalizing a scaled quaternion gives unit length."""
        q = quaternion(torch.tensor([2.0, 0.0, 0.0, 0.0]))
        result = quaternion_normalize(q)
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
        assert torch.allclose(result.wxyz, expected, atol=1e-5)

    def test_normalize_random_quaternion(self):
        """Normalized quaternion has unit norm."""
        q = quaternion(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        result = quaternion_normalize(q)
        norm = torch.linalg.norm(result.wxyz)
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-5)

    def test_double_normalize(self):
        """Normalizing twice gives the same result."""
        q = quaternion(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        result1 = quaternion_normalize(q)
        result2 = quaternion_normalize(result1)
        assert torch.allclose(result1.wxyz, result2.wxyz, atol=1e-5)

    def test_batch(self):
        """Batched normalization."""
        q = quaternion(torch.randn(10, 4))
        result = quaternion_normalize(q)
        assert result.wxyz.shape == (10, 4)
        # Check all have unit norm
        norms = torch.linalg.norm(result.wxyz, dim=-1)
        assert torch.allclose(norms, torch.ones(10), atol=1e-5)

    def test_preserves_direction(self):
        """Normalization preserves quaternion direction."""
        q = quaternion(torch.tensor([3.0, 4.0, 0.0, 0.0]))
        result = quaternion_normalize(q)
        # Should be [0.6, 0.8, 0, 0] (3/5, 4/5)
        expected = torch.tensor([0.6, 0.8, 0.0, 0.0])
        assert torch.allclose(result.wxyz, expected, atol=1e-5)


class TestQuaternionNormalizeShape:
    """Tests for quaternion_normalize shape handling."""

    def test_single_quaternion(self):
        """Single quaternion (4,) input."""
        q = quaternion(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        result = quaternion_normalize(q)
        assert result.wxyz.shape == (4,)

    def test_batch(self):
        """Batch of quaternions (B, 4)."""
        q = quaternion(torch.randn(10, 4))
        result = quaternion_normalize(q)
        assert result.wxyz.shape == (10, 4)

    def test_image_shape(self):
        """Image-like shape (H, W, 4)."""
        q = quaternion(torch.randn(64, 64, 4))
        result = quaternion_normalize(q)
        assert result.wxyz.shape == (64, 64, 4)

    def test_multi_batch_shape(self):
        """Multi-batch shape (B, C, H, W, 4)."""
        q = quaternion(torch.randn(2, 3, 16, 16, 4))
        result = quaternion_normalize(q)
        assert result.wxyz.shape == (2, 3, 16, 16, 4)

    def test_invalid_last_dim(self):
        """Raise error if last dimension is not 4."""
        with pytest.raises(ValueError, match="last dimension 4"):
            quaternion_normalize(quaternion(torch.randn(10, 3)))


class TestQuaternionNormalizeGradients:
    """Tests for quaternion_normalize gradient computation."""

    def test_gradcheck(self):
        """Gradient check w.r.t. q."""
        q = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda a: quaternion_normalize(Quaternion(wxyz=a)).wxyz,
            (q,),
            eps=1e-6,
            atol=1e-4,
        )

    def test_gradcheck_batch(self):
        """Gradient check with batch."""
        q = torch.randn(3, 5, 4, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda a: quaternion_normalize(Quaternion(wxyz=a)).wxyz,
            (q,),
            eps=1e-6,
            atol=1e-4,
        )


class TestQuaternionNormalizeDtypes:
    """Tests for quaternion_normalize with different data types."""

    def test_float32(self):
        """Works with float32."""
        q = quaternion(torch.randn(10, 4, dtype=torch.float32))
        result = quaternion_normalize(q)
        assert result.wxyz.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        q = quaternion(torch.randn(10, 4, dtype=torch.float64))
        result = quaternion_normalize(q)
        assert result.wxyz.dtype == torch.float64

    def test_bfloat16(self):
        """Works with bfloat16."""
        q = quaternion(torch.randn(10, 4, dtype=torch.bfloat16))
        result = quaternion_normalize(q)
        assert result.wxyz.dtype == torch.bfloat16

    def test_float16(self):
        """Works with float16."""
        q = quaternion(torch.randn(10, 4, dtype=torch.float16))
        result = quaternion_normalize(q)
        assert result.wxyz.dtype == torch.float16


class TestQuaternionApply:
    """Tests for quaternion_apply."""

    def test_identity_rotation(self):
        """Identity quaternion returns original point."""
        identity = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        point = torch.tensor([1.0, 2.0, 3.0])
        result = quaternion_apply(identity, point)
        assert torch.allclose(result, point, atol=1e-5)

    def test_90_deg_around_x(self):
        """90-degree rotation around x-axis maps y to z."""
        # q = [cos(45), sin(45), 0, 0]
        q = quaternion(
            torch.tensor(
                [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]
            )
        )
        point = torch.tensor([0.0, 1.0, 0.0])
        result = quaternion_apply(q, point)
        expected = torch.tensor([0.0, 0.0, 1.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_90_deg_around_y(self):
        """90-degree rotation around y-axis maps z to x."""
        # q = [cos(45), 0, sin(45), 0]
        q = quaternion(
            torch.tensor(
                [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]
            )
        )
        point = torch.tensor([0.0, 0.0, 1.0])
        result = quaternion_apply(q, point)
        expected = torch.tensor([1.0, 0.0, 0.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_90_deg_around_z(self):
        """90-degree rotation around z-axis maps x to y."""
        # q = [cos(45), 0, 0, sin(45)]
        q = quaternion(
            torch.tensor(
                [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
            )
        )
        point = torch.tensor([1.0, 0.0, 0.0])
        result = quaternion_apply(q, point)
        expected = torch.tensor([0.0, 1.0, 0.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_180_deg_around_z(self):
        """180-degree rotation around z-axis inverts x and y."""
        # q = [0, 0, 0, 1]
        q = quaternion(torch.tensor([0.0, 0.0, 0.0, 1.0]))
        point = torch.tensor([1.0, 2.0, 3.0])
        result = quaternion_apply(q, point)
        expected = torch.tensor([-1.0, -2.0, 3.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_batch(self):
        """Batched rotation."""
        # Generate random unit quaternions
        q_raw = torch.randn(10, 4)
        q = quaternion(q_raw / torch.linalg.norm(q_raw, dim=-1, keepdim=True))
        point = torch.randn(10, 3)
        result = quaternion_apply(q, point)
        assert result.shape == (10, 3)

    def test_broadcast_single_q_multiple_points(self):
        """Single quaternion applied to multiple points."""
        q = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        points = torch.randn(10, 3)
        result = quaternion_apply(q, points)
        assert result.shape == (10, 3)
        assert torch.allclose(result, points, atol=1e-5)

    def test_broadcast_multiple_q_single_point(self):
        """Multiple quaternions applied to single point."""
        # All identity quaternions
        q = quaternion(torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 10))
        point = torch.tensor([1.0, 2.0, 3.0])
        result = quaternion_apply(q, point)
        assert result.shape == (10, 3)
        expected = point.unsqueeze(0).expand(10, -1)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_broadcast_2d(self):
        """Broadcasting 2D batch dimensions."""
        q = quaternion(torch.randn(5, 1, 4))
        q = quaternion(
            q.wxyz / torch.linalg.norm(q.wxyz, dim=-1, keepdim=True)
        )
        point = torch.randn(1, 3, 3)
        result = quaternion_apply(q, point)
        assert result.shape == (5, 3, 3)

    def test_preserves_length(self):
        """Rotation preserves vector length."""
        q_raw = torch.randn(4)
        q = quaternion(q_raw / torch.linalg.norm(q_raw))
        point = torch.randn(3)
        result = quaternion_apply(q, point)
        assert torch.allclose(
            torch.linalg.norm(result), torch.linalg.norm(point), atol=1e-5
        )

    def test_inverse_rotation(self):
        """Inverse quaternion reverses the rotation."""
        q_raw = torch.randn(4)
        q = quaternion(q_raw / torch.linalg.norm(q_raw))
        q_inv = quaternion_inverse(q)
        point = torch.randn(3)
        rotated = quaternion_apply(q, point)
        back = quaternion_apply(q_inv, rotated)
        assert torch.allclose(back, point, atol=1e-5)


class TestQuaternionApplyShape:
    """Tests for quaternion_apply shape handling."""

    def test_single(self):
        """Single quaternion and point."""
        q = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        point = torch.tensor([1.0, 2.0, 3.0])
        result = quaternion_apply(q, point)
        assert result.shape == (3,)

    def test_batch(self):
        """Batch of quaternions and points."""
        q = quaternion(torch.randn(10, 4))
        point = torch.randn(10, 3)
        result = quaternion_apply(q, point)
        assert result.shape == (10, 3)

    def test_image_shape(self):
        """Image-like shape."""
        q = quaternion(torch.randn(64, 64, 4))
        point = torch.randn(64, 64, 3)
        result = quaternion_apply(q, point)
        assert result.shape == (64, 64, 3)

    def test_invalid_q_dim(self):
        """Raise error if q last dimension is not 4."""
        with pytest.raises(RuntimeError, match="last dimension 4"):
            q = quaternion(torch.randn(10, 4))
            point = torch.randn(10, 3)
            # Manually call with wrong shape to bypass Python check
            torch.ops.torchscience.quaternion_apply(torch.randn(10, 3), point)

    def test_invalid_point_dim(self):
        """Raise error if point last dimension is not 3."""
        with pytest.raises(RuntimeError, match="last dimension 3"):
            q = quaternion(torch.randn(10, 4))
            torch.ops.torchscience.quaternion_apply(q.wxyz, torch.randn(10, 4))


class TestQuaternionApplyGradients:
    """Tests for quaternion_apply gradient computation."""

    def test_gradcheck_q(self):
        """Gradient check w.r.t. q."""
        q = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
        point = torch.randn(5, 3, dtype=torch.float64)
        assert gradcheck(
            lambda a: quaternion_apply(Quaternion(wxyz=a), point),
            (q,),
            eps=1e-6,
            atol=1e-4,
        )

    def test_gradcheck_point(self):
        """Gradient check w.r.t. point."""
        q = torch.randn(5, 4, dtype=torch.float64)
        point = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda p: quaternion_apply(Quaternion(wxyz=q), p),
            (point,),
            eps=1e-6,
            atol=1e-4,
        )

    def test_gradcheck_both(self):
        """Gradient check w.r.t. both inputs."""
        q = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
        point = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda a, p: quaternion_apply(Quaternion(wxyz=a), p),
            (q, point),
            eps=1e-6,
            atol=1e-4,
        )

    def test_gradcheck_broadcast(self):
        """Gradient check with broadcasting."""
        q = torch.randn(3, 1, 4, dtype=torch.float64, requires_grad=True)
        point = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda a, p: quaternion_apply(Quaternion(wxyz=a), p),
            (q, point),
            eps=1e-6,
            atol=1e-4,
        )


class TestQuaternionApplyDtypes:
    """Tests for quaternion_apply with different data types."""

    def test_float32(self):
        """Works with float32."""
        q = quaternion(torch.randn(10, 4, dtype=torch.float32))
        point = torch.randn(10, 3, dtype=torch.float32)
        result = quaternion_apply(q, point)
        assert result.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        q = quaternion(torch.randn(10, 4, dtype=torch.float64))
        point = torch.randn(10, 3, dtype=torch.float64)
        result = quaternion_apply(q, point)
        assert result.dtype == torch.float64

    def test_bfloat16(self):
        """Works with bfloat16."""
        q = quaternion(torch.randn(10, 4, dtype=torch.bfloat16))
        point = torch.randn(10, 3, dtype=torch.bfloat16)
        result = quaternion_apply(q, point)
        assert result.dtype == torch.bfloat16

    def test_float16(self):
        """Works with float16."""
        q = quaternion(torch.randn(10, 4, dtype=torch.float16))
        point = torch.randn(10, 3, dtype=torch.float16)
        result = quaternion_apply(q, point)
        assert result.dtype == torch.float16

    def test_dtype_mismatch_error(self):
        """Raises error when dtypes don't match."""
        q = quaternion(torch.randn(10, 4, dtype=torch.float32))
        point = torch.randn(10, 3, dtype=torch.float64)
        with pytest.raises(RuntimeError, match="same dtype"):
            quaternion_apply(q, point)


class TestQuaternionToMatrix:
    """Tests for quaternion_to_matrix."""

    def test_identity(self):
        """Identity quaternion gives identity matrix."""
        q = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        R = quaternion_to_matrix(q)
        expected = torch.eye(3)
        assert torch.allclose(R, expected, atol=1e-5)

    def test_180_deg_around_z(self):
        """180-degree rotation around z-axis: diag([-1, -1, 1])."""
        # q = [0, 0, 0, 1] represents 180 degrees around z
        q = quaternion(torch.tensor([0.0, 0.0, 0.0, 1.0]))
        R = quaternion_to_matrix(q)
        expected = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))
        assert torch.allclose(R, expected, atol=1e-5)

    def test_180_deg_around_x(self):
        """180-degree rotation around x-axis: diag([1, -1, -1])."""
        # q = [0, 1, 0, 0] represents 180 degrees around x
        q = quaternion(torch.tensor([0.0, 1.0, 0.0, 0.0]))
        R = quaternion_to_matrix(q)
        expected = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
        assert torch.allclose(R, expected, atol=1e-5)

    def test_180_deg_around_y(self):
        """180-degree rotation around y-axis: diag([-1, 1, -1])."""
        # q = [0, 0, 1, 0] represents 180 degrees around y
        q = quaternion(torch.tensor([0.0, 0.0, 1.0, 0.0]))
        R = quaternion_to_matrix(q)
        expected = torch.diag(torch.tensor([-1.0, 1.0, -1.0]))
        assert torch.allclose(R, expected, atol=1e-5)

    def test_90_deg_around_z(self):
        """90-degree rotation around z-axis."""
        # q = [cos(45), 0, 0, sin(45)]
        q = quaternion(
            torch.tensor(
                [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
            )
        )
        R = quaternion_to_matrix(q)
        # Expected: [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        expected = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        assert torch.allclose(R, expected, atol=1e-5)

    def test_batch(self):
        """Batched quaternion to matrix conversion."""
        q = quaternion(torch.randn(10, 4))
        R = quaternion_to_matrix(q)
        assert R.shape == (10, 3, 3)

    def test_orthogonality(self):
        """R @ R.T = I for unit quaternions."""
        q_raw = torch.randn(10, 4)
        q = quaternion(q_raw / torch.linalg.norm(q_raw, dim=-1, keepdim=True))
        R = quaternion_to_matrix(q)
        RRT = torch.bmm(R, R.transpose(-1, -2))
        expected = torch.eye(3).unsqueeze(0).expand(10, -1, -1)
        assert torch.allclose(RRT, expected, atol=1e-5)

    def test_determinant_one(self):
        """det(R) = 1 for unit quaternions."""
        q_raw = torch.randn(10, 4)
        q = quaternion(q_raw / torch.linalg.norm(q_raw, dim=-1, keepdim=True))
        R = quaternion_to_matrix(q)
        dets = torch.linalg.det(R)
        expected = torch.ones(10)
        assert torch.allclose(dets, expected, atol=1e-5)

    def test_consistency_with_apply(self):
        """Matrix rotation should match quaternion_apply."""
        # Create a random unit quaternion
        q_raw = torch.randn(4)
        q = quaternion(q_raw / torch.linalg.norm(q_raw))
        point = torch.randn(3)

        # Rotate with quaternion_apply
        rotated_quat = quaternion_apply(q, point)

        # Rotate with matrix
        R = quaternion_to_matrix(q)
        rotated_mat = R @ point

        assert torch.allclose(rotated_quat, rotated_mat, atol=1e-5)

    def test_negative_quaternion_same_matrix(self):
        """q and -q produce the same rotation matrix."""
        q_raw = torch.randn(4)
        q_raw = q_raw / torch.linalg.norm(q_raw)
        q = quaternion(q_raw)
        q_neg = quaternion(-q_raw)

        R = quaternion_to_matrix(q)
        R_neg = quaternion_to_matrix(q_neg)

        assert torch.allclose(R, R_neg, atol=1e-5)


class TestQuaternionToMatrixShape:
    """Tests for quaternion_to_matrix shape handling."""

    def test_single_quaternion(self):
        """Single quaternion (4,) input gives (3, 3) output."""
        q = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        R = quaternion_to_matrix(q)
        assert R.shape == (3, 3)

    def test_batch(self):
        """Batch of quaternions (B, 4) gives (B, 3, 3) output."""
        q = quaternion(torch.randn(10, 4))
        R = quaternion_to_matrix(q)
        assert R.shape == (10, 3, 3)

    def test_image_shape(self):
        """Image-like shape (H, W, 4) gives (H, W, 3, 3) output."""
        q = quaternion(torch.randn(64, 64, 4))
        R = quaternion_to_matrix(q)
        assert R.shape == (64, 64, 3, 3)

    def test_multi_batch_shape(self):
        """Multi-batch shape (B, C, H, W, 4) gives (B, C, H, W, 3, 3) output."""
        q = quaternion(torch.randn(2, 3, 16, 16, 4))
        R = quaternion_to_matrix(q)
        assert R.shape == (2, 3, 16, 16, 3, 3)

    def test_invalid_last_dim(self):
        """Raise error if last dimension is not 4."""
        with pytest.raises(ValueError, match="last dimension 4"):
            quaternion_to_matrix(quaternion(torch.randn(10, 3)))


class TestQuaternionToMatrixGradients:
    """Tests for quaternion_to_matrix gradient computation."""

    def test_gradcheck(self):
        """Gradient check w.r.t. q."""
        q = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda a: quaternion_to_matrix(Quaternion(wxyz=a)),
            (q,),
            eps=1e-6,
            atol=1e-4,
        )

    def test_gradcheck_batch(self):
        """Gradient check with batch."""
        q = torch.randn(3, 5, 4, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda a: quaternion_to_matrix(Quaternion(wxyz=a)),
            (q,),
            eps=1e-6,
            atol=1e-4,
        )

    def test_gradcheck_single(self):
        """Gradient check for single quaternion."""
        q = torch.randn(4, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda a: quaternion_to_matrix(Quaternion(wxyz=a)),
            (q,),
            eps=1e-6,
            atol=1e-4,
        )


class TestQuaternionToMatrixDtypes:
    """Tests for quaternion_to_matrix with different data types."""

    def test_float32(self):
        """Works with float32."""
        q = quaternion(torch.randn(10, 4, dtype=torch.float32))
        R = quaternion_to_matrix(q)
        assert R.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        q = quaternion(torch.randn(10, 4, dtype=torch.float64))
        R = quaternion_to_matrix(q)
        assert R.dtype == torch.float64

    def test_bfloat16(self):
        """Works with bfloat16."""
        q = quaternion(torch.randn(10, 4, dtype=torch.bfloat16))
        R = quaternion_to_matrix(q)
        assert R.dtype == torch.bfloat16

    def test_float16(self):
        """Works with float16."""
        q = quaternion(torch.randn(10, 4, dtype=torch.float16))
        R = quaternion_to_matrix(q)
        assert R.dtype == torch.float16
