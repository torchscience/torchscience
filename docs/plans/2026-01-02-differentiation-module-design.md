# torchscience.differentiation Module Design

## Overview

The `differentiation` module provides finite difference operators for numerical differentiation of discrete data and grid-based PDE methods. It complements PyTorch's autograd by handling cases where autograd isn't available (discrete measurements, non-differentiable functions).

## Use Cases

1. **Numerical derivatives for discrete data** — sensor measurements, sampled functions, non-differentiable operations
2. **Grid-based PDE methods** — spatial discretization for method-of-lines, finite difference schemes

## Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| API style | Stencils + convenience functions | Stencils for PDE work (reuse, inspection), convenience for casual use |
| Stencil representation | Sparse (offsets, coeffs) | Natural mental model, memory-efficient, converts to dense for execution |
| Dimensionality | Full n-D from start | Supports mixed partials, anisotropic operators |
| Boundaries | Padding modes + explicit stencils + truncation | Flexibility for different use cases |
| Grid spacing | Scalar or per-dim tuple | Covers uniform and anisotropic grids; non-uniform deferred |
| Default accuracy | 2nd-order | Matches numpy.gradient, smallest stencil, easy upgrade to 4th |
| Derivative orders | Arbitrary | Fornberg's algorithm for any order |

## Core Data Structure

```python
@tensorclass
class FiniteDifferenceStencil:
    """N-dimensional finite difference stencil.

    Attributes
    ----------
    offsets : Tensor
        Integer offsets, shape (n_points, ndim). Each row is an offset vector.
        Example for 2D 5-point Laplacian: [[-1,0], [1,0], [0,-1], [0,1], [0,0]]
    coeffs : Tensor
        Coefficients, shape (n_points,). Matches offsets row-by-row.
        Example: [1, 1, 1, 1, -4] (before dividing by dx^2)
    derivative : Tuple[int, ...]
        Derivative order per dimension. (2, 0) means d^2/dx^2, (1, 1) means d^2/dxdy.
    accuracy : int
        Accuracy order of the stencil.
    """
    offsets: Tensor   # (n_points, ndim), int64
    coeffs: Tensor    # (n_points,), float
    derivative: Tuple[int, ...]
    accuracy: int
```

Key features:
- Sparse representation is canonical
- `to_dense()` method converts to convolution kernel (cached)
- Supports batch dimensions for multiple stencils
- `derivative` tuple enables mixed partials

## Stencil Construction

### Primary Constructor

```python
def finite_difference_stencil(
    derivative: Union[int, Tuple[int, ...]],
    accuracy: int = 2,
    kind: str = "central",  # "central", "forward", "backward"
    offsets: Optional[Tensor] = None,  # Custom offsets (advanced)
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> FiniteDifferenceStencil:
    """Generate finite difference stencil for arbitrary derivative.

    Uses Fornberg's algorithm to compute coefficients for given offsets.
    """
```

Examples:
- `derivative=2` -> 1D second derivative
- `derivative=(1, 1)` -> 2D mixed partial
- Custom `offsets` for compact schemes, upwind, etc.

### Pre-built Stencils

```python
def laplacian_stencil(ndim: int, accuracy: int = 2) -> FiniteDifferenceStencil:
    """Standard Laplacian (sum of second derivatives)."""

def biharmonic_stencil(ndim: int, accuracy: int = 2) -> FiniteDifferenceStencil:
    """Biharmonic operator nabla^4."""

def gradient_stencils(ndim: int, accuracy: int = 2) -> Tuple[FiniteDifferenceStencil, ...]:
    """Tuple of first-derivative stencils, one per dimension."""
```

## Stencil Application

```python
def apply_stencil(
    stencil: FiniteDifferenceStencil,
    field: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Optional[Tuple[int, ...]] = None,  # Default: trailing dims
    boundary: str = "replicate",  # "zeros", "reflect", "replicate", "circular", "valid"
    boundary_stencils: Optional[Tuple[FiniteDifferenceStencil, ...]] = None,
) -> Tensor:
    """Apply finite difference stencil to field."""
```

Execution strategy:
1. Convert sparse stencil to dense kernel (cached)
2. Scale coefficients by dx
3. Use `torch.nn.functional.conv1d/conv2d/conv3d`
4. Apply boundary_stencils at edges if provided

## Convenience Functions

### Scalar Field Operators

```python
def derivative(
    field: Tensor, dim: int, order: int = 1,
    dx: float = 1.0, accuracy: int = 2, kind: str = "central",
    boundary: str = "replicate",
) -> Tensor:
    """Derivative of arbitrary order along a single dimension."""

def gradient(
    field: Tensor, dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Optional[Tuple[int, ...]] = None, accuracy: int = 2,
    boundary: str = "replicate",
) -> Tensor:
    """Gradient (vector of first partials).

    Input: (..., *spatial_dims)
    Output: (..., ndim, *spatial_dims)
    """

def laplacian(
    field: Tensor, dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Optional[Tuple[int, ...]] = None, accuracy: int = 2,
    boundary: str = "replicate",
) -> Tensor:
    """Laplacian (sum of second partials). Same shape as input."""

def hessian(
    field: Tensor, dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Optional[Tuple[int, ...]] = None, accuracy: int = 2,
    boundary: str = "replicate",
) -> Tensor:
    """Hessian matrix.

    Input: (..., *spatial_dims)
    Output: (..., ndim, ndim, *spatial_dims)
    """

def biharmonic(
    field: Tensor, dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Optional[Tuple[int, ...]] = None, accuracy: int = 2,
    boundary: str = "replicate",
) -> Tensor:
    """Biharmonic nabla^4. Same shape as input."""
```

### Vector Field Operators

Convention: vector component dimension sits directly before spatial dimensions (channels-first).

```python
def divergence(
    vector_field: Tensor, dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Optional[Tuple[int, ...]] = None, accuracy: int = 2,
    boundary: str = "replicate",
) -> Tensor:
    """Divergence of vector field.

    Input: (..., ndim, *spatial_dims)
    Output: (..., *spatial_dims)
    """

def curl(
    vector_field: Tensor, dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Optional[Tuple[int, ...]] = None, accuracy: int = 2,
    boundary: str = "replicate",
) -> Tensor:
    """Curl of vector field (3D only).

    Input: (..., 3, nx, ny, nz)
    Output: (..., 3, nx, ny, nz)
    """

def jacobian(
    vector_field: Tensor, dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Optional[Tuple[int, ...]] = None, accuracy: int = 2,
    boundary: str = "replicate",
) -> Tensor:
    """Jacobian matrix of vector field.

    Input: (..., m, *spatial_dims)
    Output: (..., m, ndim, *spatial_dims)
    """
```

## Richardson Extrapolation

```python
def richardson_extrapolation(
    f: Callable[[float], Tensor],
    h: float,
    order: int = 2,
    ratio: float = 2.0,
    levels: int = 2,
) -> Tensor:
    """Improve finite difference accuracy via Richardson extrapolation.

    Parameters
    ----------
    f : callable
        Function f(h) -> Tensor computing finite difference at step size h.
    h : float
        Initial step size.
    order : int
        Leading error order of base method.
    ratio : float
        Step size reduction ratio between levels.
    levels : int
        Number of extrapolation levels.

    Returns
    -------
    Tensor
        Extrapolated result with improved accuracy.
    """
```

## File Structure

```
src/torchscience/differentiation/
    __init__.py                    # Public API exports
    _stencil.py                    # FiniteDifferenceStencil class
    _construction.py               # finite_difference_stencil, Fornberg's algorithm
    _apply.py                      # apply_stencil
    _scalar_operators.py           # derivative, gradient, laplacian, hessian, biharmonic
    _vector_operators.py           # divergence, curl, jacobian
    _richardson.py                 # richardson_extrapolation
    _exceptions.py                 # StencilError, BoundaryError

tests/torchscience/differentiation/
    test__stencil.py
    test__construction.py
    test__apply.py
    test__scalar_operators.py
    test__vector_operators.py
    test__richardson.py
```

## Implementation Notes

- **Pure Python with PyTorch ops** — no C++ kernels needed. `conv1d/conv2d/conv3d` handles execution.
- **Fornberg's algorithm** — standard method for computing finite difference coefficients for arbitrary derivative order and offset patterns.
- **Caching** — stencil `to_dense()` result should be cached to avoid recomputation.
- **Autograd compatibility** — all operations use standard PyTorch ops, so gradients flow through automatically.

## PyTorch Integration

| Feature | Support |
|---------|---------|
| autograd | Yes - via PyTorch conv ops |
| torch.compile | Yes - standard ops |
| autocast | Yes - conv handles dtype |
| vmap | Yes - batched conv |
| complex | Yes - PyTorch conv supports complex |
| meta | Yes - shape inference works |

## Future Extensions

- **Non-uniform grids** — coordinate-based API: `gradient(y, x=coordinates)`
- **Compact schemes** — higher accuracy with smaller stencils via implicit methods
- **GPU-optimized stencils** — custom CUDA kernels for very large fields
