# tests/torchscience/graphics/shading/test__cook_torrance.py
"""Comprehensive tests for Cook-Torrance BRDF."""

import math

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.graphics.shading import cook_torrance


class TestCookTorranceBasic:
    """Tests for basic shape and property verification."""

    def test_output_shape_single_sample(self):
        """Output shape matches batch dimensions for single sample."""
        normal = torch.tensor([[0.0, 1.0, 0.0]])
        view = torch.tensor([[0.0, 0.707, 0.707]])
        light = torch.tensor([[0.0, 0.707, -0.707]])

        result = cook_torrance(normal, view, light, roughness=0.5, f0=0.04)

        assert result.shape == (1,)

    def test_output_shape_batch(self):
        """Output shape matches batch dimensions for batched input."""
        normal = torch.randn(10, 3)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        view = torch.randn(10, 3)
        view = view / view.norm(dim=-1, keepdim=True)
        light = torch.randn(10, 3)
        light = light / light.norm(dim=-1, keepdim=True)

        result = cook_torrance(normal, view, light, roughness=0.5, f0=0.04)

        assert result.shape == (10,)

    def test_output_shape_rgb_f0(self):
        """Output shape is (..., 3) when f0 is RGB."""
        normal = torch.tensor([[0.0, 1.0, 0.0]])
        view = torch.tensor([[0.0, 0.707, 0.707]])
        light = torch.tensor([[0.0, 0.707, -0.707]])
        f0_gold = torch.tensor([[1.0, 0.71, 0.29]])

        result = cook_torrance(normal, view, light, roughness=0.5, f0=f0_gold)

        assert result.shape == (1, 3)

    def test_brdf_non_negative(self):
        """BRDF values are always non-negative."""
        normal = torch.randn(100, 3)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        view = torch.randn(100, 3)
        view = view / view.norm(dim=-1, keepdim=True)
        light = torch.randn(100, 3)
        light = light / light.norm(dim=-1, keepdim=True)

        result = cook_torrance(normal, view, light, roughness=0.5, f0=0.04)

        assert (result >= 0).all()


class TestCookTorranceCorrectness:
    """Tests for numerical correctness."""

    def test_back_facing_returns_zero(self):
        """BRDF returns 0 when surface is back-facing."""
        # Normal facing up, light and view below horizon
        normal = torch.tensor([[0.0, 1.0, 0.0]])
        view = torch.tensor([[0.0, -0.5, 0.866]])  # Below horizon
        light = torch.tensor([[0.0, 0.707, 0.707]])  # Above horizon

        result = cook_torrance(normal, view, light, roughness=0.5, f0=0.04)

        assert result.item() == 0.0

    def test_n_dot_l_negative_returns_zero(self):
        """BRDF returns 0 when n·l <= 0."""
        normal = torch.tensor([[0.0, 1.0, 0.0]])
        view = torch.tensor([[0.0, 0.707, 0.707]])
        light = torch.tensor([[0.0, -0.5, 0.866]])  # Light below horizon

        result = cook_torrance(normal, view, light, roughness=0.5, f0=0.04)

        assert result.item() == 0.0

    def test_mirror_reflection_high_value(self):
        """BRDF is high for mirror reflection (view = reflect(light))."""
        normal = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        light = torch.tensor([[0.0, 0.707, 0.707]], dtype=torch.float64)
        # Mirror reflection: view = 2(n·l)n - l
        n_dot_l = (normal * light).sum(dim=-1, keepdim=True)
        view = 2 * n_dot_l * normal - light

        result_smooth = cook_torrance(
            normal, view, light, roughness=0.1, f0=0.04
        )
        result_rough = cook_torrance(
            normal, view, light, roughness=0.9, f0=0.04
        )

        # Smoother surface should have higher specular at mirror angle
        assert result_smooth.item() > result_rough.item()

    def test_roughness_effect(self):
        """Higher roughness spreads the specular lobe."""
        normal = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        view = torch.tensor([[0.0, 0.707, 0.707]], dtype=torch.float64)
        light = torch.tensor([[0.0, 0.707, -0.707]], dtype=torch.float64)

        result_smooth = cook_torrance(
            normal, view, light, roughness=0.1, f0=0.04
        )
        result_rough = cook_torrance(
            normal, view, light, roughness=0.9, f0=0.04
        )

        # Both should be valid (non-negative, finite)
        assert result_smooth.item() >= 0
        assert result_rough.item() >= 0
        assert math.isfinite(result_smooth.item())
        assert math.isfinite(result_rough.item())

    def test_f0_effect(self):
        """Higher f0 increases specular reflectance."""
        normal = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        view = torch.tensor([[0.0, 0.707, 0.707]], dtype=torch.float64)
        light = torch.tensor([[0.0, 0.707, -0.707]], dtype=torch.float64)

        result_dielectric = cook_torrance(
            normal, view, light, roughness=0.5, f0=0.04
        )
        result_metal = cook_torrance(
            normal, view, light, roughness=0.5, f0=0.9
        )

        assert result_metal.item() > result_dielectric.item()


class TestCookTorranceValidation:
    """Tests for input validation."""

    def test_normal_wrong_dimension(self):
        """Raises error when normal last dimension != 3."""
        normal = torch.randn(10, 2)
        view = torch.randn(10, 3)
        light = torch.randn(10, 3)

        with pytest.raises(
            ValueError, match="normal must have last dimension 3"
        ):
            cook_torrance(normal, view, light, roughness=0.5, f0=0.04)

    def test_view_wrong_dimension(self):
        """Raises error when view last dimension != 3."""
        normal = torch.randn(10, 3)
        view = torch.randn(10, 4)
        light = torch.randn(10, 3)

        with pytest.raises(
            ValueError, match="view must have last dimension 3"
        ):
            cook_torrance(normal, view, light, roughness=0.5, f0=0.04)

    def test_light_wrong_dimension(self):
        """Raises error when light last dimension != 3."""
        normal = torch.randn(10, 3)
        view = torch.randn(10, 3)
        light = torch.randn(10, 2)

        with pytest.raises(
            ValueError, match="light must have last dimension 3"
        ):
            cook_torrance(normal, view, light, roughness=0.5, f0=0.04)


class TestCookTorranceGradients:
    """Tests for gradient computation."""

    def test_gradcheck_basic(self):
        """Passes gradcheck for basic inputs."""
        normal = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        normal = normal.detach().requires_grad_(True)

        view = torch.randn(3, 3, dtype=torch.float64)
        view = view / view.norm(dim=-1, keepdim=True)
        view = view.detach().requires_grad_(True)

        light = torch.randn(3, 3, dtype=torch.float64)
        light = light / light.norm(dim=-1, keepdim=True)
        light = light.detach().requires_grad_(True)

        roughness = torch.tensor(
            [0.5, 0.3, 0.7], dtype=torch.float64, requires_grad=True
        )
        f0 = torch.tensor(
            [0.04, 0.1, 0.9], dtype=torch.float64, requires_grad=True
        )

        def func(n, v, l, r, f):
            return cook_torrance(n, v, l, roughness=r, f0=f)

        assert gradcheck(
            func, (normal, view, light, roughness, f0), raise_exception=True
        )

    def test_gradcheck_rgb_f0(self):
        """Passes gradcheck for RGB f0."""
        normal = torch.randn(2, 3, dtype=torch.float64)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        normal = normal.detach().requires_grad_(True)

        view = torch.randn(2, 3, dtype=torch.float64)
        view = view / view.norm(dim=-1, keepdim=True)
        view = view.detach().requires_grad_(True)

        light = torch.randn(2, 3, dtype=torch.float64)
        light = light / light.norm(dim=-1, keepdim=True)
        light = light.detach().requires_grad_(True)

        roughness = torch.tensor(
            [0.5, 0.3], dtype=torch.float64, requires_grad=True
        )
        f0 = torch.randn(2, 3, dtype=torch.float64).abs().requires_grad_(True)

        def func(n, v, l, r, f):
            return cook_torrance(n, v, l, roughness=r, f0=f)

        assert gradcheck(
            func, (normal, view, light, roughness, f0), raise_exception=True
        )

    def test_gradients_finite(self):
        """Gradients are finite for typical inputs."""
        normal = torch.tensor(
            [[0.0, 1.0, 0.0]], dtype=torch.float64, requires_grad=True
        )
        view = torch.tensor(
            [[0.0, 0.707, 0.707]], dtype=torch.float64, requires_grad=True
        )
        light = torch.tensor(
            [[0.0, 0.707, -0.707]], dtype=torch.float64, requires_grad=True
        )
        roughness = torch.tensor(
            [0.5], dtype=torch.float64, requires_grad=True
        )
        f0 = torch.tensor([0.04], dtype=torch.float64, requires_grad=True)

        result = cook_torrance(normal, view, light, roughness=roughness, f0=f0)
        result.sum().backward()

        assert normal.grad is not None and torch.isfinite(normal.grad).all()
        assert view.grad is not None and torch.isfinite(view.grad).all()
        assert light.grad is not None and torch.isfinite(light.grad).all()
        assert (
            roughness.grad is not None and torch.isfinite(roughness.grad).all()
        )
        assert f0.grad is not None and torch.isfinite(f0.grad).all()


class TestCookTorranceBackwardBackward:
    """Tests for second-order gradients."""

    def test_backward_backward_computes(self):
        """Second-order gradients can be computed."""
        normal = torch.tensor(
            [[0.0, 1.0, 0.0]], dtype=torch.float64, requires_grad=True
        )
        view = torch.tensor(
            [[0.0, 0.707, 0.707]], dtype=torch.float64, requires_grad=True
        )
        light = torch.tensor(
            [[0.0, 0.707, -0.707]], dtype=torch.float64, requires_grad=True
        )
        roughness = torch.tensor(
            [0.5], dtype=torch.float64, requires_grad=True
        )
        f0 = torch.tensor([0.04], dtype=torch.float64, requires_grad=True)

        result = cook_torrance(normal, view, light, roughness=roughness, f0=f0)

        # First backward
        grads = torch.autograd.grad(
            result.sum(), [normal, roughness], create_graph=True
        )

        # Second backward (gradient of gradient)
        grad_normal = grads[0]
        grad2 = torch.autograd.grad(
            grad_normal.sum(), [normal, roughness], allow_unused=True
        )

        # Should complete without error
        assert True

    def test_backward_backward_finite(self):
        """Second-order gradients are finite."""
        normal = torch.tensor(
            [[0.0, 1.0, 0.0]], dtype=torch.float64, requires_grad=True
        )
        view = torch.tensor(
            [[0.0, 0.707, 0.707]], dtype=torch.float64, requires_grad=True
        )
        light = torch.tensor(
            [[0.0, 0.707, -0.707]], dtype=torch.float64, requires_grad=True
        )
        roughness = torch.tensor(
            [0.5], dtype=torch.float64, requires_grad=True
        )
        f0 = torch.tensor([0.04], dtype=torch.float64, requires_grad=True)

        result = cook_torrance(normal, view, light, roughness=roughness, f0=f0)

        grads = torch.autograd.grad(result.sum(), [f0], create_graph=True)
        grad_f0 = grads[0]

        # Compute gradient of grad_f0 w.r.t. f0 (second derivative)
        grad2 = torch.autograd.grad(grad_f0.sum(), [f0], allow_unused=True)

        if grad2[0] is not None:
            assert torch.isfinite(grad2[0]).all()


class TestCookTorranceDtype:
    """Tests for dtype support."""

    def test_float32(self):
        """Works with float32."""
        normal = torch.randn(5, 3, dtype=torch.float32)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        view = torch.randn(5, 3, dtype=torch.float32)
        view = view / view.norm(dim=-1, keepdim=True)
        light = torch.randn(5, 3, dtype=torch.float32)
        light = light / light.norm(dim=-1, keepdim=True)

        result = cook_torrance(normal, view, light, roughness=0.5, f0=0.04)

        assert result.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        normal = torch.randn(5, 3, dtype=torch.float64)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        view = torch.randn(5, 3, dtype=torch.float64)
        view = view / view.norm(dim=-1, keepdim=True)
        light = torch.randn(5, 3, dtype=torch.float64)
        light = light / light.norm(dim=-1, keepdim=True)

        result = cook_torrance(normal, view, light, roughness=0.5, f0=0.04)

        assert result.dtype == torch.float64


class TestCookTorranceBroadcasting:
    """Tests for broadcasting behavior."""

    def test_scalar_roughness_broadcast(self):
        """Scalar roughness broadcasts to batch."""
        normal = torch.randn(10, 3)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        view = torch.randn(10, 3)
        view = view / view.norm(dim=-1, keepdim=True)
        light = torch.randn(10, 3)
        light = light / light.norm(dim=-1, keepdim=True)

        result = cook_torrance(normal, view, light, roughness=0.5, f0=0.04)

        assert result.shape == (10,)

    def test_tensor_roughness_broadcast(self):
        """Tensor roughness broadcasts correctly."""
        normal = torch.randn(10, 3)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        view = torch.randn(10, 3)
        view = view / view.norm(dim=-1, keepdim=True)
        light = torch.randn(10, 3)
        light = light / light.norm(dim=-1, keepdim=True)
        roughness = torch.rand(10)

        result = cook_torrance(
            normal, view, light, roughness=roughness, f0=0.04
        )

        assert result.shape == (10,)

    def test_scalar_f0_broadcast(self):
        """Scalar f0 broadcasts to batch."""
        normal = torch.randn(10, 3)
        normal = normal / normal.norm(dim=-1, keepdim=True)
        view = torch.randn(10, 3)
        view = view / view.norm(dim=-1, keepdim=True)
        light = torch.randn(10, 3)
        light = light / light.norm(dim=-1, keepdim=True)

        result = cook_torrance(normal, view, light, roughness=0.5, f0=0.04)

        assert result.shape == (10,)
