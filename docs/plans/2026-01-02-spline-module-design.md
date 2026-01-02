# torchscience.spline Module Design

**Date:** 2026-01-02
**Status:** Draft

## Overview

The `torchscience.spline` module provides differentiable spline interpolation for PyTorch tensors. It supports general-purpose interpolation, differentiable path representation, and neural network integration.

## Use Cases

1. **General-purpose interpolation** — Fit smooth curves/surfaces to scattered or gridded data
2. **Differentiable path representation** — Parameterized curves for trajectory optimization, motion planning
3. **Neural network integration** — Learnable activation functions, normalizing flows, continuous-time models

## Design Decisions

| Aspect | Decision |
|--------|----------|
| Spline types | Cubic splines + B-splines |
| Dimensionality | Full N-D (curves, surfaces, volumes) |
| API style | tensorclass data containers + standalone functions |
| Boundary conditions | Natural, clamped, not-a-knot, periodic, custom |
| Backend | Pure PyTorch initially, C++ kernels later if needed |
| Autograd | Full second-order + vmap compatibility |
| Extrapolation | Configurable ("error" default, "clamp", "extrapolate") |
| B-spline knots | Learnable (requires_grad support) |

## Module Structure

```
src/torchscience/spline/
├── __init__.py
├── _cubic_spline.py      # CubicSpline tensorclass + functions
├── _b_spline.py          # BSpline tensorclass + functions
├── _basis.py             # B-spline basis function evaluation
└── _tridiagonal.py       # Thomas algorithm for cubic spline fitting
```

## Data Types

### CubicSpline

```python
from tensordict.tensorclass import tensorclass

@tensorclass
class CubicSpline:
    """Piecewise cubic polynomial interpolant."""
    knots: Tensor           # (n_knots,) - breakpoints
    coefficients: Tensor    # (n_segments, 4, *value_shape) - [a, b, c, d] per segment
    boundary: str           # "natural" | "clamped" | "not_a_knot" | "periodic" | "custom"
    extrapolate: str        # "error" | "clamp" | "extrapolate"
```

### BSpline

```python
@tensorclass
class BSpline:
    """B-spline curve."""
    knots: Tensor           # (n_knots,)
    control_points: Tensor  # (n_control, *value_shape)
    degree: int             # polynomial degree (typically 3)
    extrapolate: str        # "error" | "clamp" | "extrapolate"
```

### BSplineSurface

```python
@tensorclass
class BSplineSurface:
    """Tensor product B-spline surface."""
    knots: tuple[Tensor, Tensor]  # (knots_u, knots_v)
    control_points: Tensor        # (n_u, n_v, *value_shape)
    degree: tuple[int, int]       # (degree_u, degree_v)
    extrapolate: str
```

### BSplineVolume

```python
@tensorclass
class BSplineVolume:
    """Tensor product B-spline volume."""
    knots: tuple[Tensor, Tensor, Tensor]
    control_points: Tensor        # (n_u, n_v, n_w, *value_shape)
    degree: tuple[int, int, int]
    extrapolate: str
```

## Functional API

### Cubic Spline Functions

```python
# Fitting
def cubic_spline_fit(
    x: Tensor,                    # (n_points,)
    y: Tensor,                    # (n_points, *value_shape)
    boundary: str = "not_a_knot", # "natural" | "clamped" | "not_a_knot" | "periodic" | "custom"
    boundary_values: Tensor | None = None,  # for "clamped"/"custom": (2, *value_shape)
    extrapolate: str = "error",
) -> CubicSpline: ...

# Evaluation
def cubic_spline_evaluate(spline: CubicSpline, t: Tensor) -> Tensor: ...
def cubic_spline_derivative(spline: CubicSpline, t: Tensor, order: int = 1) -> Tensor: ...
def cubic_spline_integral(spline: CubicSpline, a: float, b: float) -> Tensor: ...

# Convenience (fit + evaluate in one call)
def cubic_spline_interpolate(x: Tensor, y: Tensor, t: Tensor, **kwargs) -> Tensor: ...
```

### B-Spline Functions

```python
# Construction
def b_spline(
    knots: Tensor,
    control_points: Tensor,
    degree: int = 3,
    extrapolate: str = "error",
) -> BSpline: ...

# Fitting to data
def b_spline_fit(
    x: Tensor, y: Tensor,
    n_control: int,               # number of control points
    degree: int = 3,
    knots: Tensor | None = None,  # optional custom knots
) -> BSpline: ...

# Evaluation
def b_spline_evaluate(spline: BSpline, t: Tensor) -> Tensor: ...
def b_spline_derivative(spline: BSpline, t: Tensor, order: int = 1) -> Tensor: ...
def b_spline_basis(knots: Tensor, t: Tensor, degree: int) -> Tensor: ...  # (n_basis, n_query)
```

### N-D Surface/Volume Functions

```python
# Surface evaluation
def b_spline_surface_evaluate(
    surface: BSplineSurface,
    u: Tensor, v: Tensor,        # query coordinates
) -> Tensor: ...

def b_spline_surface_derivative(
    surface: BSplineSurface,
    u: Tensor, v: Tensor,
    order_u: int = 0, order_v: int = 0,  # partial derivative orders
) -> Tensor: ...

# Fitting gridded data
def b_spline_surface_fit(
    u: Tensor, v: Tensor,        # grid coordinates (n_u,), (n_v,)
    values: Tensor,              # (n_u, n_v, *value_shape)
    n_control: tuple[int, int],
    degree: tuple[int, int] = (3, 3),
) -> BSplineSurface: ...
```

## Algorithms

### Cubic Spline Fitting (Thomas Algorithm)

Solve tridiagonal system `A @ m = rhs` for second derivatives m, then compute polynomial coefficients.

```python
def _solve_tridiagonal(diag: Tensor, upper: Tensor, lower: Tensor, rhs: Tensor) -> Tensor:
    """Thomas algorithm - O(n) tridiagonal solver, fully differentiable."""
    # Forward elimination + back substitution
    # Batched over value_shape dimensions
```

Boundary condition modifications:
- **Natural:** m_0 = m_n = 0 (known, reduce system size)
- **Clamped:** Extra equations from known first derivatives
- **Not-a-knot:** Third derivative continuity at x_1 and x_{n-1}
- **Periodic:** Wrap-around structure (Sherman-Morrison for efficient solve)

### B-Spline Basis (Cox-de Boor Recursion)

```python
def b_spline_basis(knots: Tensor, t: Tensor, degree: int) -> Tensor:
    """Evaluate all non-zero B-spline basis functions at query points."""
    # Iterative Cox-de Boor: build degree-0, then 1, ..., up to degree
    # Returns (n_query, degree+1) for the non-zero basis functions
    # Differentiable w.r.t. both knots and t
```

### Derivative Computation

- **Cubic splines:** Symbolic differentiation of polynomial coefficients
- **B-splines:** Cox-de Boor derivative formula (derivative is degree-1 B-spline with differenced control points)

### N-D Tensor Product Evaluation

Tensor product evaluation: `S(u,v) = sum_ij N_i(u) M_j(v) P_ij`

Separable structure allows evaluating each dimension independently then combining.

## Autograd & vmap Compatibility

### Gradient Flow Paths

1. **Through evaluation:** grad L / grad t (query points)
2. **Through data points:** grad L / grad y (input values, flows through tridiagonal solve)
3. **Through knots (B-spline):** grad L / grad knots (learnable knot placement)
4. **Through control points:** grad L / grad P (learnable B-splines)

### Implementation Strategy

All core functions written as pure functions (no in-place ops) to ensure vmap/jacrev/jacfwd compatibility.

```python
from torch.func import vmap, jacrev, jacfwd

# Example: batched Jacobian of spline w.r.t. control points
jacobian = jacrev(b_spline_evaluate, argnums=0)
batched_jacobian = vmap(jacobian)
```

### Tridiagonal Solve Gradient

The backward pass for `_solve_tridiagonal` solves another tridiagonal system (adjoint method) with the same O(n) complexity as forward.

## Testing Requirements

### Test Structure

```
tests/torchscience/spline/
├── test__cubic_spline.py
│   ├── test_fit_natural_boundary
│   ├── test_fit_clamped_boundary
│   ├── test_fit_not_a_knot_boundary
│   ├── test_fit_periodic_boundary
│   ├── test_evaluate_interpolates_data_points
│   ├── test_derivative_analytical_comparison
│   ├── test_integral_analytical_comparison
│   ├── test_extrapolate_error_raises
│   ├── test_extrapolate_clamp
│   ├── test_gradcheck / test_gradgradcheck
│   └── test_scipy_comparison
├── test__b_spline.py
│   ├── test_basis_partition_of_unity
│   ├── test_evaluate_endpoints
│   ├── test_learnable_knots_gradient
│   ├── test_learnable_control_points_gradient
│   └── test_vmap_compatibility
└── test__b_spline_surface.py
    ├── test_tensor_product_separability
    └── test_partial_derivatives
```

### Testing Standards

Every function must pass:
```python
torch.autograd.gradcheck(fn, inputs, check_batched_grad=True)
torch.autograd.gradgradcheck(fn, inputs)
```

vmap compatibility:
```python
from torch.func import vmap
vmap(fn)(batched_inputs)  # must work without error
```

### scipy Comparison

Numerical validation against `scipy.interpolate.CubicSpline` and `scipy.interpolate.BSpline` with tight tolerances (~1e-10 for float64).

## Error Handling

```python
class SplineError(Exception): ...
class ExtrapolationError(SplineError): ...  # query outside bounds with extrapolate="error"
class KnotError(SplineError): ...           # non-monotonic, insufficient knots
class DegreeError(SplineError): ...         # degree > n_knots - 1
```

Validation in fit functions:
- x must be strictly monotonically increasing
- len(x) >= 2 for cubic, >= degree+1 for B-spline
- y.shape[0] == len(x)
- boundary_values shape matches when required
