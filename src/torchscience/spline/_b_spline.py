"""B-spline basis function evaluation using Cox-de Boor recursion."""

from typing import Optional

import torch
from tensordict.tensorclass import tensorclass
from torch import Tensor

from torchscience.spline._exceptions import (
    DegreeError,
    ExtrapolationError,
    KnotError,
)

__all__ = [
    "BSpline",
    "b_spline_basis",
    "b_spline_evaluate",
    "b_spline_fit",
]


@tensorclass
class BSpline:
    """B-spline representation with knots and control points.

    Attributes
    ----------
    knots : Tensor
        Knot vector, shape (n_knots,). Non-decreasing.
    control_points : Tensor
        Control points, shape (n_control, *y_dim) where n_control = n_knots - degree - 1
    degree : int
        Polynomial degree (stored as metadata, not tensor)
    extrapolate : str
        How to handle out-of-domain queries: "error", "clamp", "extrapolate"
    """

    knots: Tensor
    control_points: Tensor
    degree: int
    extrapolate: str


def b_spline_basis(
    t: Tensor,
    knots: Tensor,
    degree: int,
    i: Optional[int] = None,
) -> Tensor:
    """
    Evaluate B-spline basis functions using Cox-de Boor recursion.

    Parameters
    ----------
    t : Tensor
        Evaluation points, shape (*query_shape)
    knots : Tensor
        Knot vector, shape (n_knots,). Must be non-decreasing.
    degree : int
        Polynomial degree (0=constant, 1=linear, 2=quadratic, 3=cubic)
    i : int, optional
        If specified, return only the i-th basis function. Otherwise return all.

    Returns
    -------
    basis : Tensor
        If i is None: shape (*query_shape, n_basis) where n_basis = n_knots - degree - 1
        If i is specified: shape (*query_shape)

    Raises
    ------
    DegreeError
        If degree is negative or too high for the given knot count.
    KnotError
        If knots are not non-decreasing.

    Notes
    -----
    For degree 0:
        B_{i,0}(t) = 1 if t_i <= t < t_{i+1}, else 0

    For degree k > 0:
        B_{i,k}(t) = ((t - t_i) / (t_{i+k} - t_i)) * B_{i,k-1}(t)
                   + ((t_{i+k+1} - t) / (t_{i+k+1} - t_{i+1})) * B_{i+1,k-1}(t)

    Division by zero (0/0) is handled as 0 (when knot intervals are zero).

    The implementation uses dynamic programming (bottom-up evaluation) rather
    than naive recursion for efficiency.
    """
    n_knots = knots.shape[0]

    # Validate degree
    if degree < 0:
        raise DegreeError(f"Degree must be non-negative, got {degree}")
    if n_knots < degree + 2:
        raise DegreeError(
            f"Need at least {degree + 2} knots for degree {degree}, got {n_knots}"
        )

    # Validate knots are non-decreasing
    if not torch.all(knots[1:] >= knots[:-1]):
        raise KnotError("Knots must be non-decreasing")

    # Number of basis functions
    n_basis = n_knots - degree - 1

    # Handle scalar input
    is_scalar = t.dim() == 0
    if is_scalar:
        t = t.unsqueeze(0)

    # Store original query shape and flatten for computation
    query_shape = t.shape
    t_flat = t.flatten()  # (n_points,)
    n_points = t_flat.shape[0]

    # Initialize degree 0 basis functions
    # B_{j,0}(t) = 1 if t_j <= t < t_{j+1}, else 0
    # We need (n_knots - 1) degree-0 basis functions for the recursion
    # Shape: (n_points, n_knots - 1)

    # Use >= for left boundary and < for right boundary
    # Handle rightmost point special case: t == knots[-1] should be in the last interval
    t_expanded = t_flat.unsqueeze(-1)  # (n_points, 1)
    knots_left = knots[:-1].unsqueeze(0)  # (1, n_knots - 1)
    knots_right = knots[1:].unsqueeze(0)  # (1, n_knots - 1)

    # Standard interval check: t_i <= t < t_{i+1}
    basis_0 = ((t_expanded >= knots_left) & (t_expanded < knots_right)).to(
        dtype=knots.dtype
    )

    # Handle rightmost point: if t == knots[-1], include in last interval
    # Find where t equals the last knot
    at_right_boundary = t_flat == knots[-1]
    if at_right_boundary.any():
        # Find the last non-empty interval (where knots differ)
        # For each rightmost point, set the last interval with span > 0 to 1
        # But for simplicity, set the rightmost interval to 1
        # Actually, we need to find the rightmost interval that exists
        # For uniform knots, this is simply the last one
        # For repeated knots at the end, we need to find the last interval with positive span
        last_valid_interval = n_knots - 2
        for j in range(n_knots - 2, -1, -1):
            if knots[j] < knots[j + 1]:
                last_valid_interval = j
                break
        basis_0[at_right_boundary, last_valid_interval] = 1.0

    # Dynamic programming: build up from degree 0 to target degree
    # At each level k, we compute B_{j,k} for j = 0, 1, ..., n_knots - k - 2
    # The number of basis functions at degree k is: n_knots - k - 1

    basis_current = basis_0  # Shape: (n_points, n_knots - 1)

    for k in range(1, degree + 1):
        # Number of basis functions at this degree
        n_basis_k = n_knots - k - 1

        # Allocate new basis tensor
        basis_next = torch.zeros(
            n_points, n_basis_k, dtype=knots.dtype, device=knots.device
        )

        for j in range(n_basis_k):
            # Left term: ((t - t_j) / (t_{j+k} - t_j)) * B_{j,k-1}(t)
            denom_left = knots[j + k] - knots[j]
            if denom_left.abs() > 0:
                alpha_left = (t_flat - knots[j]) / denom_left
                left_term = alpha_left * basis_current[:, j]
            else:
                # 0/0 case: treat as 0
                left_term = torch.zeros(
                    n_points, dtype=knots.dtype, device=knots.device
                )

            # Right term: ((t_{j+k+1} - t) / (t_{j+k+1} - t_{j+1})) * B_{j+1,k-1}(t)
            denom_right = knots[j + k + 1] - knots[j + 1]
            if denom_right.abs() > 0:
                alpha_right = (knots[j + k + 1] - t_flat) / denom_right
                right_term = alpha_right * basis_current[:, j + 1]
            else:
                # 0/0 case: treat as 0
                right_term = torch.zeros(
                    n_points, dtype=knots.dtype, device=knots.device
                )

            basis_next[:, j] = left_term + right_term

        basis_current = basis_next

    # basis_current now has shape (n_points, n_basis)
    # Reshape back to (*query_shape, n_basis)
    result = basis_current.view(*query_shape, n_basis)

    # Handle scalar input: remove the leading dimension
    if is_scalar:
        result = result.squeeze(0)

    # If i is specified, return only the i-th basis function
    if i is not None:
        if i < 0 or i >= n_basis:
            raise IndexError(
                f"Basis index {i} out of range [0, {n_basis - 1}]"
            )
        # Select the i-th basis function
        result = result[..., i]

    return result


def b_spline_evaluate(
    spline: BSpline,
    t: Tensor,
) -> Tensor:
    """
    Evaluate a B-spline at query points.

    Parameters
    ----------
    spline : BSpline
        B-spline with knots and control points
    t : Tensor
        Query points, shape (*query_shape)

    Returns
    -------
    y : Tensor
        Evaluated values, shape (*query_shape, *y_dim)

    Raises
    ------
    ExtrapolationError
        If any query point is outside the spline domain and
        spline.extrapolate == 'error'

    Notes
    -----
    B-spline evaluation: y(t) = sum_i B_{i,k}(t) * c_i
    where B_{i,k} are basis functions and c_i are control points.
    """
    knots = spline.knots
    control_points = spline.control_points
    degree = spline.degree
    extrapolate = spline.extrapolate

    # Check if t is scalar (0-d tensor)
    is_scalar = t.dim() == 0
    if is_scalar:
        t = t.unsqueeze(0)

    # Store original query shape
    query_shape = t.shape
    t_flat = t.flatten()

    # Get domain bounds
    # For B-splines, the domain is [knots[degree], knots[n_knots - degree - 1]]
    # For clamped knots (repeated at ends), this is [knots[0], knots[-1]]
    t_min = knots[degree]
    t_max = knots[-(degree + 1)]

    # Handle extrapolation modes
    if extrapolate == "error":
        if torch.any(t_flat < t_min) or torch.any(t_flat > t_max):
            raise ExtrapolationError(
                f"Query points outside spline domain [{t_min.item()}, {t_max.item()}]"
            )
    elif extrapolate == "clamp":
        t_flat = torch.clamp(t_flat, t_min, t_max)
    # "extrapolate" mode: no special handling, basis functions will handle it

    # Reshape t_flat back to compute basis
    # Compute basis functions: shape (*query_shape, n_control)
    basis = b_spline_basis(t_flat.view(*query_shape), knots, degree)

    # control_points shape: (n_control, *y_dim) or (n_control,) for 1D
    # basis shape: (*query_shape, n_control)
    # Result shape: (*query_shape, *y_dim)

    if control_points.dim() == 1:
        # 1D control points: simple weighted sum
        # basis: (*query_shape, n_control), control_points: (n_control,)
        # einsum: ...i, i -> ...
        y = torch.einsum("...i,i->...", basis, control_points)
    else:
        # Multi-dimensional control points
        # basis: (*query_shape, n_control), control_points: (n_control, *y_dim)
        # We need: y[..., j] = sum_i basis[..., i] * control_points[i, j]
        # This is a matrix multiplication along the control point axis
        y_dim = control_points.shape[1:]
        n_control = control_points.shape[0]

        # Flatten basis to (n_query, n_control) and control_points to (n_control, n_values)
        n_query = basis[..., 0].numel()
        basis_flat = basis.reshape(n_query, n_control)
        control_flat = control_points.reshape(n_control, -1)

        # Matrix multiply: (n_query, n_control) @ (n_control, n_values) -> (n_query, n_values)
        y_flat = torch.mm(basis_flat, control_flat)

        # Reshape back to (*query_shape, *y_dim)
        y = y_flat.view(*query_shape, *y_dim)

    # Handle scalar input: return scalar output (or with y_dim)
    if is_scalar:
        y = y.squeeze(0)

    return y


def b_spline_fit(
    x: Tensor,
    y: Tensor,
    degree: int = 3,
    n_knots: Optional[int] = None,
    knots: Optional[Tensor] = None,
    extrapolate: str = "error",
) -> BSpline:
    """
    Fit a B-spline to data points using least squares.

    Parameters
    ----------
    x : Tensor
        Data x-coordinates, shape (n_points,)
    y : Tensor
        Data y-values, shape (n_points,) or (n_points, *y_dim)
    degree : int
        Polynomial degree (default 3 for cubic)
    n_knots : int, optional
        Number of interior knots (uniformly spaced). Mutually exclusive with knots.
    knots : Tensor, optional
        Explicit knot vector. If not provided, creates clamped uniform knots.
    extrapolate : str
        Extrapolation mode for returned BSpline

    Returns
    -------
    spline : BSpline
        Fitted B-spline

    Raises
    ------
    ValueError
        If both n_knots and knots are provided, or if inputs are invalid.
    KnotError
        If knots are not non-decreasing.
    DegreeError
        If degree is negative or too high for the given knot count.

    Notes
    -----
    Solves the least squares problem: min ||B @ c - y||^2
    where B is the collocation matrix (basis functions at data points).
    """
    n_points = x.shape[0]

    # Validate inputs
    if n_points < 2:
        raise ValueError(f"Need at least 2 data points, got {n_points}")

    if degree < 0:
        raise DegreeError(f"Degree must be non-negative, got {degree}")

    if n_knots is not None and knots is not None:
        raise ValueError("Cannot specify both n_knots and knots")

    # Generate knots if not provided
    if knots is None:
        # Determine number of interior knots
        if n_knots is None:
            # Default: min(n_points // 2, 10)
            n_knots = min(n_points // 2, 10)

        # Create clamped uniform knot vector
        # Total knots = n_interior + 2*(degree+1)
        # For n_knots interior knots in [x_min, x_max]:
        # - (degree+1) copies of x_min at start
        # - n_knots uniformly spaced interior knots
        # - (degree+1) copies of x_max at end
        x_min = x.min()
        x_max = x.max()

        if n_knots > 0:
            # Interior knots uniformly spaced between x_min and x_max
            interior_knots = torch.linspace(
                x_min, x_max, n_knots + 2, dtype=x.dtype, device=x.device
            )[1:-1]
        else:
            interior_knots = torch.tensor([], dtype=x.dtype, device=x.device)

        # Build clamped knot vector
        boundary_left = x_min.expand(degree + 1)
        boundary_right = x_max.expand(degree + 1)
        knots = torch.cat([boundary_left, interior_knots, boundary_right])

    # Validate knots
    if not torch.all(knots[1:] >= knots[:-1]):
        raise KnotError("Knots must be non-decreasing")

    n_knots_total = knots.shape[0]
    if n_knots_total < degree + 2:
        raise DegreeError(
            f"Need at least {degree + 2} knots for degree {degree}, got {n_knots_total}"
        )

    # Number of control points = n_knots - degree - 1
    n_control = n_knots_total - degree - 1

    if n_points < n_control:
        raise ValueError(
            f"Need at least {n_control} data points for {n_control} control points, got {n_points}"
        )

    # Build collocation matrix B: B[i, j] = B_j(x[i])
    # Shape: (n_points, n_control)
    B = b_spline_basis(x, knots, degree)

    # Handle y dimensions
    if y.dim() == 1:
        y_flat = y.unsqueeze(-1)  # (n_points, 1)
        is_1d = True
    else:
        y_flat = y.reshape(n_points, -1)  # (n_points, n_values)
        is_1d = False

    n_values = y_flat.shape[1]

    # Solve least squares: min ||B @ c - y||^2
    # Using torch.linalg.lstsq: returns c such that ||B @ c - y||^2 is minimized
    solution = torch.linalg.lstsq(B, y_flat)
    control_points = solution.solution  # (n_control, n_values)

    # Reshape control points back to original y shape
    if is_1d:
        control_points = control_points.squeeze(-1)  # (n_control,)
    else:
        y_dim = y.shape[1:]
        control_points = control_points.reshape(n_control, *y_dim)

    return BSpline(
        knots=knots,
        control_points=control_points,
        degree=degree,
        extrapolate=extrapolate,
        batch_size=[],
    )
