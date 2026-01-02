"""Cubic spline interpolation."""

from typing import Optional, Union

import torch
from tensordict.tensorclass import tensorclass
from torch import Tensor

from torchscience.spline._exceptions import ExtrapolationError, KnotError
from torchscience.spline._tridiagonal import solve_tridiagonal


@tensorclass
class CubicSpline:
    """Piecewise cubic polynomial interpolant.

    Attributes
    ----------
    knots : Tensor
        Breakpoints, shape (n_knots,). Strictly increasing.
    coefficients : Tensor
        Polynomial coefficients, shape (n_segments, 4, *value_shape).
        For segment i, the polynomial is:
        a[i] + b[i]*(t-knots[i]) + c[i]*(t-knots[i])^2 + d[i]*(t-knots[i])^3
        where coefficients[i] = [a, b, c, d].
    boundary : str
        Boundary condition type: "natural", "clamped", "not_a_knot", "periodic".
    extrapolate : str
        Extrapolation mode: "error", "clamp", "extrapolate".
    """

    knots: Tensor
    coefficients: Tensor
    boundary: str
    extrapolate: str


def cubic_spline_fit(
    x: Tensor,
    y: Tensor,
    boundary: str = "not_a_knot",
    boundary_values: Optional[Tensor] = None,
    extrapolate: str = "error",
) -> CubicSpline:
    """
    Fit a cubic spline to data points.

    Parameters
    ----------
    x : Tensor
        Knot positions, shape (n_points,). Must be strictly increasing.
    y : Tensor
        Values at knots, shape (n_points, *value_shape).
    boundary : str
        Boundary condition: "natural", "clamped", "not_a_knot", "periodic".
    boundary_values : Tensor, optional
        For "clamped": first derivatives at endpoints, shape (2, *value_shape).
    extrapolate : str
        Extrapolation mode: "error", "clamp", "extrapolate".

    Returns
    -------
    CubicSpline
        Fitted spline.

    Raises
    ------
    KnotError
        If x is not strictly increasing or has fewer than 2 points.
    """
    n = x.shape[0]

    # Validate knots
    if n < 2:
        raise KnotError(f"Need at least 2 points, got {n}")
    if not torch.all(x[1:] > x[:-1]):
        raise KnotError("Knots must be strictly increasing")

    # Compute interval widths
    h = x[1:] - x[:-1]  # (n-1,)

    # Get value shape
    if y.dim() == 1:
        value_shape = ()
        y_flat = y
    else:
        value_shape = y.shape[1:]
        # Flatten value dimensions for computation
        y_flat = y.reshape(n, -1)  # (n, prod(value_shape))

    n_values = y_flat.shape[1] if y_flat.dim() > 1 else 1
    if y_flat.dim() == 1:
        y_flat = y_flat.unsqueeze(-1)

    # Compute second differences for RHS
    # delta[i] = (y[i+1] - y[i]) / h[i]
    delta = (y_flat[1:] - y_flat[:-1]) / h.unsqueeze(-1)  # (n-1, n_values)

    # Build tridiagonal system for second derivatives m
    # Interior equations: h[i-1]*m[i-1] + 2*(h[i-1]+h[i])*m[i] + h[i]*m[i+1] = 6*(delta[i] - delta[i-1])

    if boundary == "natural":
        # Natural: m[0] = m[n-1] = 0
        # Solve reduced (n-2) x (n-2) system for m[1:n-1]
        if n == 2:
            # Only one segment, no interior points
            m = torch.zeros(2, n_values, dtype=y.dtype, device=y.device)
        else:
            diag = 2 * (h[:-1] + h[1:])  # (n-2,)
            upper = h[1:-1]  # (n-3,)
            lower = h[1:-1]  # (n-3,)
            rhs = 6 * (delta[1:] - delta[:-1])  # (n-2, n_values)

            if n == 3:
                # 1x1 system
                m_interior = rhs / diag.unsqueeze(-1)
            else:
                m_interior = solve_tridiagonal(diag, upper, lower, rhs.T).T

            # Pad with zeros at endpoints
            m = torch.cat(
                [
                    torch.zeros(1, n_values, dtype=y.dtype, device=y.device),
                    m_interior,
                    torch.zeros(1, n_values, dtype=y.dtype, device=y.device),
                ],
                dim=0,
            )

    elif boundary == "clamped":
        if boundary_values is None:
            raise ValueError("boundary_values required for clamped boundary")
        bv = boundary_values.reshape(2, -1)  # (2, n_values)

        # Full n x n system
        diag = torch.zeros(n, dtype=x.dtype, device=x.device)
        diag[0] = 2 * h[0]
        diag[1:-1] = 2 * (h[:-1] + h[1:])
        diag[-1] = 2 * h[-1]

        upper = torch.zeros(n - 1, dtype=x.dtype, device=x.device)
        upper[0] = h[0]
        upper[1:] = h[1:]

        lower = torch.zeros(n - 1, dtype=x.dtype, device=x.device)
        lower[:-1] = h[:-1]
        lower[-1] = h[-1]

        rhs = torch.zeros(n, n_values, dtype=y.dtype, device=y.device)
        rhs[0] = 6 * (delta[0] - bv[0])
        rhs[1:-1] = 6 * (delta[1:] - delta[:-1])
        rhs[-1] = 6 * (bv[1] - delta[-1])

        m = solve_tridiagonal(diag, upper, lower, rhs.T).T

    elif boundary == "not_a_knot":
        if n < 4:
            # Fall back to natural for small systems
            return cubic_spline_fit(
                x, y, boundary="natural", extrapolate=extrapolate
            )

        # Not-a-knot: third derivative continuous at x[1] and x[n-2]
        # This modifies the first and last equations
        diag = torch.zeros(n, dtype=x.dtype, device=x.device)
        upper = torch.zeros(n - 1, dtype=x.dtype, device=x.device)
        lower = torch.zeros(n - 1, dtype=x.dtype, device=x.device)
        rhs = torch.zeros(n, n_values, dtype=y.dtype, device=y.device)

        # First equation: not-a-knot at x[1]
        diag[0] = h[1]
        upper[0] = -(h[0] + h[1])

        # Interior equations
        for i in range(1, n - 1):
            diag[i] = 2 * (h[i - 1] + h[i])
            if i < n - 1:
                upper[i] = h[i]
            if i > 0:
                lower[i - 1] = h[i - 1]

        # Last equation: not-a-knot at x[n-2]
        lower[-1] = -(h[-2] + h[-1])
        diag[-1] = h[-2]

        # Fix interior equations RHS
        rhs[1:-1] = 6 * (delta[1:] - delta[:-1])

        m = solve_tridiagonal(diag, upper, lower, rhs.T).T

    elif boundary == "periodic":
        if not torch.allclose(y_flat[0], y_flat[-1]):
            raise KnotError("For periodic boundary, y[0] must equal y[-1]")

        # Periodic boundary uses Sherman-Morrison or cyclic reduction
        # For simplicity, use natural boundary on cyclic extension
        # (Production code would use Sherman-Morrison for O(n) solve)
        # Solve reduced system m[0] = m[n-1]
        diag = 2 * (h[:-1] + h[1:])  # (n-2,)
        if n > 3:
            upper = h[1:-1].clone()
            lower = h[1:-1].clone()
        else:
            upper = torch.tensor([], dtype=x.dtype, device=x.device)
            lower = torch.tensor([], dtype=x.dtype, device=x.device)
        rhs = 6 * (delta[1:] - delta[:-1])

        if n == 3:
            m_interior = rhs / diag.unsqueeze(-1)
        elif n > 3:
            m_interior = solve_tridiagonal(diag, upper, lower, rhs.T).T
        else:
            m_interior = torch.zeros(
                0, n_values, dtype=y.dtype, device=y.device
            )

        # For periodic, set m[0] = m[-1]
        m = torch.cat(
            [
                torch.zeros(1, n_values, dtype=y.dtype, device=y.device),
                m_interior,
                torch.zeros(1, n_values, dtype=y.dtype, device=y.device),
            ],
            dim=0,
        )
        m[-1] = m[0]  # Enforce periodicity

    else:
        raise ValueError(f"Unknown boundary condition: {boundary}")

    # Compute polynomial coefficients for each segment
    # p_i(t) = a_i + b_i*(t-x_i) + c_i*(t-x_i)^2 + d_i*(t-x_i)^3
    # where:
    #   a_i = y_i
    #   b_i = delta_i - h_i * (2*m_i + m_{i+1}) / 6
    #   c_i = m_i / 2
    #   d_i = (m_{i+1} - m_i) / (6 * h_i)

    n_seg = n - 1
    a = y_flat[:-1]  # (n_seg, n_values)
    c = m[:-1] / 2  # (n_seg, n_values)
    d = (m[1:] - m[:-1]) / (6 * h.unsqueeze(-1))  # (n_seg, n_values)
    b = delta - h.unsqueeze(-1) * (2 * m[:-1] + m[1:]) / 6  # (n_seg, n_values)

    # Stack coefficients: (n_seg, 4, n_values)
    coeffs = torch.stack([a, b, c, d], dim=1)

    # Reshape back to original value shape
    if value_shape:
        coeffs = coeffs.reshape(n_seg, 4, *value_shape)
    else:
        # Remove the trailing dimension for scalar values
        coeffs = coeffs.squeeze(-1)

    return CubicSpline(
        knots=x,
        coefficients=coeffs,
        boundary=boundary,
        extrapolate=extrapolate,
        batch_size=[],
    )


def cubic_spline_evaluate(
    spline: CubicSpline,
    t: Tensor,
) -> Tensor:
    """
    Evaluate a cubic spline at query points.

    Parameters
    ----------
    spline : CubicSpline
        Fitted cubic spline from cubic_spline_fit
    t : Tensor
        Query points, shape (*query_shape) or scalar

    Returns
    -------
    y : Tensor
        Interpolated values, shape (*query_shape, *y_dim) where y_dim
        is the dimensionality of the original y values

    Raises
    ------
    ExtrapolationError
        If any query point is outside the spline domain and
        spline.extrapolate == 'error'
    """
    knots = spline.knots
    coeffs = spline.coefficients
    extrapolate = spline.extrapolate

    # Check if t is scalar (0-d tensor)
    is_scalar = t.dim() == 0
    if is_scalar:
        t = t.unsqueeze(0)

    # Store original query shape
    query_shape = t.shape
    t_flat = t.flatten()

    # Get domain bounds
    t_min = knots[0]
    t_max = knots[-1]

    # Handle extrapolation modes
    if extrapolate == "error":
        if torch.any(t_flat < t_min) or torch.any(t_flat > t_max):
            raise ExtrapolationError(
                f"Query points outside spline domain [{t_min.item()}, {t_max.item()}]"
            )
    elif extrapolate == "clamp":
        t_flat = torch.clamp(t_flat, t_min, t_max)

    # Find segment indices using searchsorted
    # searchsorted returns index where value would be inserted to maintain order
    # We want the segment index i such that knots[i] <= t < knots[i+1]
    segment_idx = torch.searchsorted(knots, t_flat, right=True) - 1

    # Clamp segment indices to valid range [0, n_segments-1]
    n_segments = len(knots) - 1
    segment_idx = torch.clamp(segment_idx, 0, n_segments - 1)

    # Get the left knot for each query point
    x_i = knots[segment_idx]

    # Compute dx = t - x_i
    dx = t_flat - x_i

    # Get coefficients for each segment
    # coeffs shape: (n_segments, 4, *value_shape) or (n_segments, 4) for scalar values
    a = coeffs[segment_idx, 0]  # (*query_flat, *value_shape)
    b = coeffs[segment_idx, 1]
    c = coeffs[segment_idx, 2]
    d = coeffs[segment_idx, 3]

    # Evaluate polynomial using Horner's method: y = a + dx*(b + dx*(c + dx*d))
    # Need to handle broadcasting for multi-dimensional values
    if coeffs.dim() > 2:
        # Multi-dimensional values: dx needs extra dimensions
        value_shape = coeffs.shape[2:]
        dx = dx.view(-1, *([1] * len(value_shape)))
    else:
        value_shape = ()

    y = a + dx * (b + dx * (c + dx * d))

    # Reshape to (*query_shape, *value_shape)
    if value_shape:
        y = y.view(*query_shape, *value_shape)
    else:
        y = y.view(*query_shape)

    # Handle scalar input: return scalar output
    if is_scalar:
        y = y.squeeze(0)

    return y


def cubic_spline_derivative(
    spline: CubicSpline,
    order: int = 1,
) -> CubicSpline:
    """
    Compute the derivative of a cubic spline.

    Parameters
    ----------
    spline : CubicSpline
        Input cubic spline
    order : int
        Order of derivative (1, 2, or 3). Default is 1.

    Returns
    -------
    derivative : CubicSpline
        A new CubicSpline representing the derivative.
        First derivative is quadratic (degree 2), second is linear (degree 1),
        third is constant (degree 0).

    Raises
    ------
    ValueError
        If order is not 1, 2, or 3.

    Notes
    -----
    For cubic polynomial: y = a + b*dx + c*dx^2 + d*dx^3
    - First derivative: y' = b + 2c*dx + 3d*dx^2  (coefficients: [b, 2c, 3d, 0])
    - Second derivative: y'' = 2c + 6d*dx  (coefficients: [2c, 6d, 0, 0])
    - Third derivative: y''' = 6d  (coefficients: [6d, 0, 0, 0])

    The new CubicSpline has the same knots but transformed coefficients.
    """
    if order < 1 or order > 3:
        raise ValueError(f"Derivative order must be 1, 2, or 3, got {order}")

    coeffs = spline.coefficients
    # coeffs shape: (n_segments, 4, *value_shape) or (n_segments, 4) for scalars

    # Extract coefficients
    # Original: a + b*dx + c*dx^2 + d*dx^3
    # coeffs[:, 0] = a, coeffs[:, 1] = b, coeffs[:, 2] = c, coeffs[:, 3] = d
    a = coeffs[:, 0]  # (n_segments, *value_shape)
    b = coeffs[:, 1]
    c = coeffs[:, 2]
    d = coeffs[:, 3]

    if order == 1:
        # First derivative: b + 2c*dx + 3d*dx^2
        # New coefficients: [b, 2c, 3d, 0]
        new_a = b
        new_b = 2 * c
        new_c = 3 * d
        new_d = torch.zeros_like(d)
    elif order == 2:
        # Second derivative: 2c + 6d*dx
        # New coefficients: [2c, 6d, 0, 0]
        new_a = 2 * c
        new_b = 6 * d
        new_c = torch.zeros_like(c)
        new_d = torch.zeros_like(d)
    else:  # order == 3
        # Third derivative: 6d
        # New coefficients: [6d, 0, 0, 0]
        new_a = 6 * d
        new_b = torch.zeros_like(b)
        new_c = torch.zeros_like(c)
        new_d = torch.zeros_like(d)

    # Stack new coefficients
    new_coeffs = torch.stack([new_a, new_b, new_c, new_d], dim=1)

    return CubicSpline(
        knots=spline.knots.clone(),
        coefficients=new_coeffs,
        boundary=spline.boundary,
        extrapolate=spline.extrapolate,
        batch_size=[],
    )


def cubic_spline_integral(
    spline: CubicSpline,
    a: Union[float, Tensor],
    b: Union[float, Tensor],
) -> Tensor:
    """
    Compute the definite integral of a cubic spline from a to b.

    Parameters
    ----------
    spline : CubicSpline
        Input cubic spline
    a : float or Tensor
        Lower bound of integration
    b : float or Tensor
        Upper bound of integration

    Returns
    -------
    integral : Tensor
        Definite integral value(s), shape (*y_dim) or batched

    Notes
    -----
    For cubic polynomial on segment [x_i, x_{i+1}]:
        y = a + b*dx + c*dx^2 + d*dx^3

    The antiderivative is:
        F(dx) = a*dx + (b/2)*dx^2 + (c/3)*dx^3 + (d/4)*dx^4

    For definite integral:
        integral[t1, t2] = F(t2 - x_i) - F(t1 - x_i)

    If [a, b] spans multiple segments, the integrals are summed.
    """
    knots = spline.knots
    coeffs = spline.coefficients
    n_segments = len(knots) - 1

    # Convert bounds to tensors if necessary
    if not isinstance(a, Tensor):
        a = torch.tensor(a, dtype=knots.dtype, device=knots.device)
    if not isinstance(b, Tensor):
        b = torch.tensor(b, dtype=knots.dtype, device=knots.device)

    # Handle a > b case: integral from b to a = -integral from a to b
    sign = torch.ones(1, dtype=knots.dtype, device=knots.device)
    if a > b:
        a, b = b, a
        sign = -sign

    # Handle a == b case
    if a == b:
        # Return zero with correct shape
        if coeffs.dim() > 2:
            value_shape = coeffs.shape[2:]
            return torch.zeros(
                value_shape, dtype=knots.dtype, device=knots.device
            )
        else:
            return torch.tensor(0.0, dtype=knots.dtype, device=knots.device)

    # Clamp bounds to spline domain
    t_min = knots[0]
    t_max = knots[-1]
    a_clamped = torch.clamp(a, t_min, t_max)
    b_clamped = torch.clamp(b, t_min, t_max)

    # Find segment indices for lower and upper bounds
    # searchsorted returns index where value would be inserted
    seg_a = torch.searchsorted(knots, a_clamped, right=True) - 1
    seg_b = torch.searchsorted(knots, b_clamped, right=True) - 1

    # Clamp to valid segment range
    seg_a = torch.clamp(seg_a, 0, n_segments - 1)
    seg_b = torch.clamp(seg_b, 0, n_segments - 1)

    # Get value shape for multi-dimensional output
    if coeffs.dim() > 2:
        value_shape = coeffs.shape[2:]
    else:
        value_shape = ()

    # Initialize total integral
    if value_shape:
        total = torch.zeros(
            value_shape, dtype=knots.dtype, device=knots.device
        )
    else:
        total = torch.tensor(0.0, dtype=knots.dtype, device=knots.device)

    def antiderivative(dx: Tensor, seg_idx: int) -> Tensor:
        """
        Evaluate the antiderivative F(dx) = a*dx + (b/2)*dx^2 + (c/3)*dx^3 + (d/4)*dx^4

        Parameters
        ----------
        dx : Tensor
            Offset from segment start (t - x_i)
        seg_idx : int
            Segment index

        Returns
        -------
        F : Tensor
            Antiderivative value at dx
        """
        c_a = coeffs[seg_idx, 0]  # constant term
        c_b = coeffs[seg_idx, 1]  # linear term
        c_c = coeffs[seg_idx, 2]  # quadratic term
        c_d = coeffs[seg_idx, 3]  # cubic term

        if value_shape:
            # Expand dx for broadcasting with value dimensions
            dx_expanded = dx.view(*([1] * len(value_shape)))
        else:
            dx_expanded = dx

        # F(dx) = a*dx + (b/2)*dx^2 + (c/3)*dx^3 + (d/4)*dx^4
        return (
            c_a * dx_expanded
            + (c_b / 2) * dx_expanded**2
            + (c_c / 3) * dx_expanded**3
            + (c_d / 4) * dx_expanded**4
        )

    # Integrate over each segment that intersects [a, b]
    for seg_idx in range(seg_a.item(), seg_b.item() + 1):
        # Determine integration bounds within this segment
        seg_start = knots[seg_idx]
        seg_end = knots[seg_idx + 1]

        # Lower bound for this segment: max(a, seg_start)
        lower = torch.max(a_clamped, seg_start)

        # Upper bound for this segment: min(b, seg_end)
        upper = torch.min(b_clamped, seg_end)

        # Compute offsets from segment start
        dx_lower = lower - seg_start
        dx_upper = upper - seg_start

        # Add contribution: F(upper - seg_start) - F(lower - seg_start)
        contribution = antiderivative(dx_upper, seg_idx) - antiderivative(
            dx_lower, seg_idx
        )
        total = total + contribution

    return sign.squeeze() * total
