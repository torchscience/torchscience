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


class TestCubicSplineEvaluate:
    def test_evaluate_at_knots(self):
        """Test that evaluating at knot points returns original y values."""
        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.sin(x * math.pi)

        spline = cubic_spline_fit(x, y)
        y_eval = cubic_spline_evaluate(spline, x)

        torch.testing.assert_close(y_eval, y, atol=1e-12, rtol=1e-12)

    def test_evaluate_between_knots(self):
        """Test that evaluating between knots interpolates smoothly."""
        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        # Use a polynomial that a cubic spline can represent exactly
        # For y = x^3, dy/dx = 3x^2, so at x=0: 0, at x=1: 3
        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**3  # Cubic polynomial

        # Use clamped boundary with correct derivatives for exact representation
        boundary_values = torch.tensor([0.0, 3.0], dtype=torch.float64)
        spline = cubic_spline_fit(
            x, y, boundary="clamped", boundary_values=boundary_values
        )

        # Query at midpoints
        x_query = torch.tensor(
            [0.1, 0.25, 0.5, 0.75, 0.9], dtype=torch.float64
        )
        y_eval = cubic_spline_evaluate(spline, x_query)

        # For a cubic polynomial with matching boundary derivatives, spline is exact
        y_expected = x_query**3
        torch.testing.assert_close(y_eval, y_expected, atol=1e-6, rtol=1e-6)

    def test_evaluate_scalar_query(self):
        """Test that a single scalar query point works."""
        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.sin(x)

        spline = cubic_spline_fit(x, y)

        # Single query point (0-d tensor)
        x_query = torch.tensor(0.5, dtype=torch.float64)
        y_eval = cubic_spline_evaluate(spline, x_query)

        # Result should also be a scalar
        assert y_eval.shape == ()
        # Value should be reasonable (between min and max of y)
        assert y_eval >= y.min() and y_eval <= y.max()

    def test_evaluate_extrapolate_error(self):
        """Test that query outside domain with extrapolate='error' raises ExtrapolationError."""
        from torchscience.spline import (
            ExtrapolationError,
            cubic_spline_evaluate,
            cubic_spline_fit,
        )

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.sin(x)

        spline = cubic_spline_fit(x, y, extrapolate="error")

        # Query below domain
        x_query_below = torch.tensor([-0.1], dtype=torch.float64)
        with pytest.raises(ExtrapolationError):
            cubic_spline_evaluate(spline, x_query_below)

        # Query above domain
        x_query_above = torch.tensor([1.1], dtype=torch.float64)
        with pytest.raises(ExtrapolationError):
            cubic_spline_evaluate(spline, x_query_above)

    def test_evaluate_extrapolate_clamp(self):
        """Test that query outside domain with extrapolate='clamp' clamps to boundary."""
        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.tensor([1.0, 2.0, 3.0, 2.5, 2.0], dtype=torch.float64)

        spline = cubic_spline_fit(x, y, extrapolate="clamp")

        # Query below domain should return value at x[0]
        x_query_below = torch.tensor([-1.0, -0.5], dtype=torch.float64)
        y_eval_below = cubic_spline_evaluate(spline, x_query_below)
        y_at_start = cubic_spline_evaluate(
            spline, torch.tensor([0.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            y_eval_below, y_at_start.expand(2), atol=1e-12, rtol=1e-12
        )

        # Query above domain should return value at x[-1]
        x_query_above = torch.tensor([1.5, 2.0], dtype=torch.float64)
        y_eval_above = cubic_spline_evaluate(spline, x_query_above)
        y_at_end = cubic_spline_evaluate(
            spline, torch.tensor([1.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            y_eval_above, y_at_end.expand(2), atol=1e-12, rtol=1e-12
        )

    def test_evaluate_extrapolate_extend(self):
        """Test that query outside domain with extrapolate='extend' extrapolates using boundary polynomial."""
        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        # Use a linear function - extrapolation should follow the line
        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = 2 * x + 1  # y = 2x + 1

        spline = cubic_spline_fit(x, y, extrapolate="extend")

        # Query outside domain
        x_query = torch.tensor([-0.5, 1.5], dtype=torch.float64)
        y_eval = cubic_spline_evaluate(spline, x_query)

        # For a linear function, extrapolation should be exact
        y_expected = 2 * x_query + 1
        torch.testing.assert_close(y_eval, y_expected, atol=1e-6, rtol=1e-6)

    def test_evaluate_multidimensional(self):
        """Test evaluation with multi-dimensional y values (e.g., 3D curve)."""
        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        # 3D curve: (sin(t), cos(t), t)
        y = torch.stack(
            [torch.sin(x * math.pi), torch.cos(x * math.pi), x], dim=-1
        )  # (5, 3)

        spline = cubic_spline_fit(x, y)

        # Evaluate at knots
        y_eval = cubic_spline_evaluate(spline, x)

        assert y_eval.shape == (5, 3)
        torch.testing.assert_close(y_eval, y, atol=1e-12, rtol=1e-12)

        # Evaluate at midpoint
        x_mid = torch.tensor([0.5], dtype=torch.float64)
        y_mid = cubic_spline_evaluate(spline, x_mid)

        assert y_mid.shape == (1, 3)

    def test_gradcheck(self):
        """Test that gradients flow through evaluation."""
        from torch.autograd import gradcheck

        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.sin(x * math.pi)

        spline = cubic_spline_fit(x, y)

        # Query points need gradients
        x_query = torch.tensor(
            [0.25, 0.5, 0.75], dtype=torch.float64, requires_grad=True
        )

        def eval_fn(xq):
            return cubic_spline_evaluate(spline, xq)

        assert gradcheck(eval_fn, (x_query,), eps=1e-6, atol=1e-4)

    def test_scipy_comparison(self):
        """Test that results match scipy.interpolate.CubicSpline."""
        scipy = pytest.importorskip("scipy")
        from scipy.interpolate import CubicSpline as ScipyCubicSpline

        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        x = torch.linspace(0, 2 * math.pi, 10, dtype=torch.float64)
        y = torch.sin(x)

        # Fit with torchscience
        spline = cubic_spline_fit(x, y, boundary="natural")

        # Fit with scipy
        scipy_spline = ScipyCubicSpline(
            x.numpy(), y.numpy(), bc_type="natural"
        )

        # Evaluate at many points
        x_query = torch.linspace(0, 2 * math.pi, 100, dtype=torch.float64)
        y_torch = cubic_spline_evaluate(spline, x_query)
        y_scipy = torch.from_numpy(scipy_spline(x_query.numpy()))

        torch.testing.assert_close(y_torch, y_scipy, atol=1e-10, rtol=1e-10)


class TestCubicSplineDerivative:
    """Tests for cubic_spline_derivative function."""

    def test_derivative_of_cubic(self):
        """Test that derivative of x^3 should be 3x^2."""
        from torchscience.spline import (
            cubic_spline_derivative,
            cubic_spline_evaluate,
            cubic_spline_fit,
        )

        # Fit x^3 with appropriate boundary conditions
        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**3

        # Use clamped boundary with correct derivatives for exact representation
        # dy/dx = 3x^2, so at x=0: 0, at x=1: 3
        boundary_values = torch.tensor([0.0, 3.0], dtype=torch.float64)
        spline = cubic_spline_fit(
            x, y, boundary="clamped", boundary_values=boundary_values
        )

        # Get derivative spline
        deriv_spline = cubic_spline_derivative(spline, order=1)

        # Evaluate derivative at test points
        x_test = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
        deriv_eval = cubic_spline_evaluate(deriv_spline, x_test)

        # Expected: 3x^2
        expected = 3 * x_test**2
        torch.testing.assert_close(deriv_eval, expected, atol=1e-6, rtol=1e-6)

    def test_derivative_of_quadratic(self):
        """Test that derivative of x^2 should be 2x."""
        from torchscience.spline import (
            cubic_spline_derivative,
            cubic_spline_evaluate,
            cubic_spline_fit,
        )

        # Fit x^2
        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**2

        # Use clamped boundary with correct derivatives
        # dy/dx = 2x, so at x=0: 0, at x=1: 2
        boundary_values = torch.tensor([0.0, 2.0], dtype=torch.float64)
        spline = cubic_spline_fit(
            x, y, boundary="clamped", boundary_values=boundary_values
        )

        # Get derivative spline
        deriv_spline = cubic_spline_derivative(spline, order=1)

        # Evaluate derivative at test points
        x_test = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
        deriv_eval = cubic_spline_evaluate(deriv_spline, x_test)

        # Expected: 2x
        expected = 2 * x_test
        torch.testing.assert_close(deriv_eval, expected, atol=1e-6, rtol=1e-6)

    def test_second_derivative(self):
        """Test that second derivative of x^3 should be 6x."""
        from torchscience.spline import (
            cubic_spline_derivative,
            cubic_spline_evaluate,
            cubic_spline_fit,
        )

        # Fit x^3
        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**3

        # Use clamped boundary with correct derivatives
        boundary_values = torch.tensor([0.0, 3.0], dtype=torch.float64)
        spline = cubic_spline_fit(
            x, y, boundary="clamped", boundary_values=boundary_values
        )

        # Get second derivative spline
        deriv_spline = cubic_spline_derivative(spline, order=2)

        # Evaluate second derivative at test points
        x_test = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
        deriv_eval = cubic_spline_evaluate(deriv_spline, x_test)

        # Expected: 6x
        expected = 6 * x_test
        torch.testing.assert_close(deriv_eval, expected, atol=1e-6, rtol=1e-6)

    def test_third_derivative(self):
        """Test that third derivative of x^3 should be 6 (constant)."""
        from torchscience.spline import (
            cubic_spline_derivative,
            cubic_spline_evaluate,
            cubic_spline_fit,
        )

        # Fit x^3
        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**3

        # Use clamped boundary with correct derivatives
        boundary_values = torch.tensor([0.0, 3.0], dtype=torch.float64)
        spline = cubic_spline_fit(
            x, y, boundary="clamped", boundary_values=boundary_values
        )

        # Get third derivative spline
        deriv_spline = cubic_spline_derivative(spline, order=3)

        # Evaluate third derivative at test points
        x_test = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
        deriv_eval = cubic_spline_evaluate(deriv_spline, x_test)

        # Expected: constant 6
        expected = torch.full_like(x_test, 6.0)
        torch.testing.assert_close(deriv_eval, expected, atol=1e-6, rtol=1e-6)

    def test_derivative_at_points(self):
        """Test derivative evaluation against analytical derivative of sin(x)."""
        from torchscience.spline import (
            cubic_spline_derivative,
            cubic_spline_evaluate,
            cubic_spline_fit,
        )

        # Fit sin(x)
        x = torch.linspace(0, 2 * math.pi, 20, dtype=torch.float64)
        y = torch.sin(x)

        spline = cubic_spline_fit(x, y, boundary="natural")

        # Get derivative spline
        deriv_spline = cubic_spline_derivative(spline, order=1)

        # Evaluate derivative at interior test points (avoid boundary effects)
        x_test = torch.linspace(
            0.5, 2 * math.pi - 0.5, 10, dtype=torch.float64
        )
        deriv_eval = cubic_spline_evaluate(deriv_spline, x_test)

        # Expected: cos(x)
        expected = torch.cos(x_test)

        # Use looser tolerance since spline approximation isn't exact
        torch.testing.assert_close(deriv_eval, expected, atol=1e-3, rtol=1e-3)

    def test_derivative_invalid_order(self):
        """Test that invalid derivative order raises ValueError."""
        from torchscience.spline import (
            cubic_spline_derivative,
            cubic_spline_fit,
        )

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**2
        spline = cubic_spline_fit(x, y)

        # Order 0 is invalid
        with pytest.raises(ValueError):
            cubic_spline_derivative(spline, order=0)

        # Order > 3 is invalid
        with pytest.raises(ValueError):
            cubic_spline_derivative(spline, order=4)

        # Negative order is invalid
        with pytest.raises(ValueError):
            cubic_spline_derivative(spline, order=-1)

    def test_derivative_preserves_knots(self):
        """Test that derivative preserves the knot vector."""
        from torchscience.spline import (
            cubic_spline_derivative,
            cubic_spline_fit,
        )

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**3
        spline = cubic_spline_fit(x, y)

        deriv_spline = cubic_spline_derivative(spline, order=1)

        torch.testing.assert_close(deriv_spline.knots, spline.knots)

    def test_derivative_multidimensional(self):
        """Test derivative with multi-dimensional y values."""
        from torchscience.spline import (
            cubic_spline_derivative,
            cubic_spline_evaluate,
            cubic_spline_fit,
        )

        # Fit a 2D curve: (t^2, t^3)
        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.stack([x**2, x**3], dim=-1)  # (5, 2)

        spline = cubic_spline_fit(x, y)
        deriv_spline = cubic_spline_derivative(spline, order=1)

        # Evaluate derivative at test points
        x_test = torch.tensor([0.5], dtype=torch.float64)
        deriv_eval = cubic_spline_evaluate(deriv_spline, x_test)

        # Expected: (2t, 3t^2) at t=0.5: (1.0, 0.75)
        expected = torch.tensor([[1.0, 0.75]], dtype=torch.float64)
        torch.testing.assert_close(deriv_eval, expected, atol=1e-4, rtol=1e-4)
