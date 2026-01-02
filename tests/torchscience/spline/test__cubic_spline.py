"""Tests for cubic spline interpolation."""

import math

import pytest
import torch


class TestCubicSplineFit:
    def test_fit_returns_cubic_spline(self):
        """Test that cubic_spline_fit returns a CubicSpline tensorclass."""
        from torchscience.spline import CubicSpline, cubic_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.sin(x)

        spline = cubic_spline_fit(x, y)

        assert isinstance(spline, CubicSpline)
        assert spline.knots.shape == (5,)
        assert spline.coefficients.shape == (4, 4)  # 4 segments, 4 coeffs each

    def test_fit_natural_boundary(self):
        """Test natural boundary conditions (zero second derivative at ends)."""
        from torchscience.spline import cubic_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**3  # Known cubic

        spline = cubic_spline_fit(x, y, boundary="natural")

        # For natural splines, second derivative at endpoints should be ~0
        # Coefficient layout: [a, b, c, d] for a + b*(t-ti) + c*(t-ti)^2 + d*(t-ti)^3
        # Second derivative at ti is 2*c
        # Check first segment at x[0]
        assert abs(spline.coefficients[0, 2].item()) < 1e-6

    def test_fit_not_a_knot_boundary(self):
        """Test not-a-knot boundary (default)."""
        from torchscience.spline import cubic_spline_fit

        x = torch.linspace(0, 2 * math.pi, 10, dtype=torch.float64)
        y = torch.sin(x)

        spline = cubic_spline_fit(x, y, boundary="not_a_knot")

        assert spline.boundary == "not_a_knot"

    def test_fit_clamped_boundary(self):
        """Test clamped boundary conditions (known first derivatives)."""
        from torchscience.spline import cubic_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**2

        # dy/dx = 2x, so at x=0: 0, at x=1: 2
        boundary_values = torch.tensor([0.0, 2.0], dtype=torch.float64)

        spline = cubic_spline_fit(
            x, y, boundary="clamped", boundary_values=boundary_values
        )

        assert spline.boundary == "clamped"

    def test_fit_periodic_boundary(self):
        """Test periodic boundary conditions."""
        from torchscience.spline import cubic_spline_fit

        x = torch.linspace(0, 2 * math.pi, 10, dtype=torch.float64)
        y = torch.sin(x)

        spline = cubic_spline_fit(x, y, boundary="periodic")

        assert spline.boundary == "periodic"

    def test_fit_validates_monotonic_knots(self):
        """Test that non-monotonic knots raise KnotError."""
        from torchscience.spline import KnotError, cubic_spline_fit

        x = torch.tensor(
            [0.0, 0.5, 0.3, 1.0], dtype=torch.float64
        )  # Not monotonic
        y = torch.tensor([0.0, 0.5, 0.3, 1.0], dtype=torch.float64)

        with pytest.raises(KnotError):
            cubic_spline_fit(x, y)

    def test_fit_validates_minimum_points(self):
        """Test that too few points raises KnotError."""
        from torchscience.spline import KnotError, cubic_spline_fit

        x = torch.tensor([0.0], dtype=torch.float64)
        y = torch.tensor([1.0], dtype=torch.float64)

        with pytest.raises(KnotError):
            cubic_spline_fit(x, y)

    def test_fit_multidimensional_values(self):
        """Test fitting with multi-dimensional y values (e.g., 3D curve)."""
        from torchscience.spline import cubic_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.stack([torch.sin(x), torch.cos(x), x], dim=-1)  # (5, 3)

        spline = cubic_spline_fit(x, y)

        # 4 segments, 4 coefficients, 3 value dimensions
        assert spline.coefficients.shape == (4, 4, 3)
