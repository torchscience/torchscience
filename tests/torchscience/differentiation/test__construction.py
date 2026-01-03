"""Tests for stencil construction functions."""

import pytest
import torch

from torchscience.differentiation import finite_difference_stencil


class TestFornbergAlgorithm:
    """Tests for Fornberg's algorithm implementation."""

    def test_first_derivative_central_accuracy_2(self):
        """Central first derivative with 2nd order accuracy."""
        stencil = finite_difference_stencil(
            derivative=1, accuracy=2, kind="central"
        )
        # Should be [-1/2, 0, 1/2]
        expected_coeffs = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64)
        expected_offsets = torch.tensor([[-1], [0], [1]])
        torch.testing.assert_close(stencil.coeffs, expected_coeffs)
        torch.testing.assert_close(stencil.offsets, expected_offsets)

    def test_second_derivative_central_accuracy_2(self):
        """Central second derivative with 2nd order accuracy."""
        stencil = finite_difference_stencil(
            derivative=2, accuracy=2, kind="central"
        )
        # Should be [1, -2, 1]
        expected_coeffs = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float64)
        torch.testing.assert_close(
            stencil.coeffs, expected_coeffs, rtol=1e-5, atol=1e-7
        )

    def test_first_derivative_forward_accuracy_1(self):
        """Forward first derivative with 1st order accuracy."""
        stencil = finite_difference_stencil(
            derivative=1, accuracy=1, kind="forward"
        )
        # Should be [-1, 1]
        expected_coeffs = torch.tensor([-1.0, 1.0], dtype=torch.float64)
        expected_offsets = torch.tensor([[0], [1]])
        torch.testing.assert_close(stencil.coeffs, expected_coeffs)
        torch.testing.assert_close(stencil.offsets, expected_offsets)

    def test_first_derivative_backward_accuracy_1(self):
        """Backward first derivative with 1st order accuracy."""
        stencil = finite_difference_stencil(
            derivative=1, accuracy=1, kind="backward"
        )
        # Should be [-1, 1] with offsets [-1, 0]
        expected_coeffs = torch.tensor([-1.0, 1.0], dtype=torch.float64)
        expected_offsets = torch.tensor([[-1], [0]])
        torch.testing.assert_close(stencil.coeffs, expected_coeffs)
        torch.testing.assert_close(stencil.offsets, expected_offsets)

    def test_fourth_derivative_central_accuracy_2(self):
        """Central fourth derivative with 2nd order accuracy."""
        stencil = finite_difference_stencil(
            derivative=4, accuracy=2, kind="central"
        )
        # Should be [1, -4, 6, -4, 1]
        expected_coeffs = torch.tensor(
            [1.0, -4.0, 6.0, -4.0, 1.0], dtype=torch.float64
        )
        torch.testing.assert_close(
            stencil.coeffs, expected_coeffs, rtol=1e-5, atol=1e-7
        )

    def test_higher_order_accuracy(self):
        """Central first derivative with 4th order accuracy."""
        stencil = finite_difference_stencil(
            derivative=1, accuracy=4, kind="central"
        )
        # Should be [1/12, -2/3, 0, 2/3, -1/12]
        expected_coeffs = torch.tensor(
            [1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12], dtype=torch.float64
        )
        torch.testing.assert_close(
            stencil.coeffs, expected_coeffs, rtol=1e-5, atol=1e-7
        )

    def test_derivative_tuple_1d(self):
        """Derivative specified as tuple for 1D."""
        stencil = finite_difference_stencil(
            derivative=(2,), accuracy=2, kind="central"
        )
        assert stencil.derivative == (2,)
        assert stencil.ndim == 1

    def test_derivative_tuple_2d_mixed(self):
        """Mixed partial derivative d^2/dxdy."""
        stencil = finite_difference_stencil(
            derivative=(1, 1), accuracy=2, kind="central"
        )
        assert stencil.derivative == (1, 1)
        assert stencil.ndim == 2
        # Mixed partial should have 4 corners
        assert stencil.n_points == 4

    def test_derivative_tuple_2d_second_x(self):
        """Second derivative in x: d^2/dx^2."""
        stencil = finite_difference_stencil(
            derivative=(2, 0), accuracy=2, kind="central"
        )
        assert stencil.derivative == (2, 0)
        assert stencil.ndim == 2
        # Should only vary in first dimension
        assert torch.all(stencil.offsets[:, 1] == 0)


class TestStencilValidation:
    """Tests for stencil validation."""

    def test_invalid_kind_raises(self):
        """Invalid kind raises error."""
        with pytest.raises(ValueError, match="kind"):
            finite_difference_stencil(derivative=1, accuracy=2, kind="invalid")

    def test_zero_derivative_raises(self):
        """Zero derivative order raises error."""
        with pytest.raises(ValueError, match="derivative"):
            finite_difference_stencil(derivative=0, accuracy=2, kind="central")

    def test_zero_accuracy_raises(self):
        """Zero accuracy raises error."""
        with pytest.raises(ValueError, match="accuracy"):
            finite_difference_stencil(derivative=1, accuracy=0, kind="central")

    def test_negative_derivative_raises(self):
        """Negative derivative raises error."""
        with pytest.raises(ValueError, match="derivative"):
            finite_difference_stencil(
                derivative=-1, accuracy=2, kind="central"
            )
