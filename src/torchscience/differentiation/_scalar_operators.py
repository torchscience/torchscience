"""Scalar field differential operators.

High-level functions for computing derivatives, gradients, Laplacians,
Hessians, and biharmonic operators on scalar fields.
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor

from torchscience.differentiation._apply import apply_stencil
from torchscience.differentiation._construction import (
    finite_difference_stencil,
)


def derivative(
    field: Tensor,
    dim: int,
    order: int = 1,
    dx: float = 1.0,
    accuracy: int = 2,
    kind: str = "central",
    boundary: str = "replicate",
) -> Tensor:
    """Compute derivative of a scalar field along a single dimension.

    Parameters
    ----------
    field : Tensor
        Input scalar field with arbitrary shape.
    dim : int
        Dimension along which to compute the derivative.
    order : int, optional
        Order of the derivative (1 for first, 2 for second, etc.). Default is 1.
    dx : float, optional
        Grid spacing. Default is 1.0.
    accuracy : int, optional
        Accuracy order of the finite difference approximation. Default is 2.
    kind : str, optional
        Stencil type: "central", "forward", or "backward". Default is "central".
    boundary : str, optional
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate".

    Returns
    -------
    Tensor
        Derivative field with the same shape as the input (unless boundary="valid").

    Examples
    --------
    >>> x = torch.linspace(0, 1, 21)
    >>> f = x**2
    >>> df = derivative(f, dim=0, order=1, dx=0.05)  # df/dx = 2x
    >>> d2f = derivative(f, dim=0, order=2, dx=0.05)  # d^2f/dx^2 = 2
    """
    # Normalize dimension to positive index
    ndim = field.ndim
    if dim < 0:
        dim = ndim + dim
    if dim < 0 or dim >= ndim:
        raise ValueError(
            f"dim {dim} out of range for tensor with {ndim} dimensions"
        )

    # Create 1D stencil
    stencil = finite_difference_stencil(
        derivative=order,
        accuracy=accuracy,
        kind=kind,
        dtype=field.dtype,
        device=field.device,
    )

    # Move the target dimension to the end, apply stencil, then move back
    # This is needed because apply_stencil operates on trailing dimensions
    perm = list(range(ndim))
    perm.remove(dim)
    perm.append(dim)

    field_permuted = field.permute(perm)
    result_permuted = apply_stencil(
        stencil, field_permuted, dx=dx, boundary=boundary
    )

    # Inverse permutation
    inv_perm = [0] * ndim
    for i, p in enumerate(perm):
        inv_perm[p] = i

    return result_permuted.permute(inv_perm)


def gradient(
    field: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
) -> Tensor:
    """Compute gradient of a scalar field.

    The gradient is the vector of partial derivatives along each spatial dimension.

    Parameters
    ----------
    field : Tensor
        Input scalar field with shape (..., *spatial_dims).
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0.
    dim : tuple of int, optional
        Spatial dimensions over which to compute gradient. Default uses all dimensions.
    accuracy : int, optional
        Accuracy order of the finite difference approximation. Default is 2.
    boundary : str, optional
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate".

    Returns
    -------
    Tensor
        Gradient field. If dim is None or specifies all dimensions, shape is
        (ndim, *field.shape). If dim specifies a subset, shape is
        (...batch_dims, len(dim), *spatial_dims) where the gradient components
        are inserted before the spatial dimensions.

    Examples
    --------
    >>> # 2D field
    >>> f = torch.randn(20, 30)
    >>> grad = gradient(f, dx=0.1)  # Shape: (2, 20, 30)

    >>> # Batched field with gradient over last 2 dimensions
    >>> f = torch.randn(10, 20, 30)
    >>> grad = gradient(f, dim=(-2, -1), dx=0.1)  # Shape: (10, 2, 20, 30)
    """
    ndim = field.ndim

    # Determine which dimensions to differentiate
    if dim is None:
        # All dimensions
        spatial_dims = tuple(range(ndim))
    else:
        spatial_dims = tuple(d if d >= 0 else ndim + d for d in dim)

    n_spatial = len(spatial_dims)

    # Handle dx
    if isinstance(dx, (int, float)):
        dx_tuple = (float(dx),) * n_spatial
    else:
        dx_tuple = tuple(float(d) for d in dx)
        if len(dx_tuple) != n_spatial:
            raise ValueError(
                f"dx has {len(dx_tuple)} elements but {n_spatial} spatial dimensions"
            )

    # Compute partial derivatives
    partials = []
    for i, d in enumerate(spatial_dims):
        partial = derivative(
            field,
            dim=d,
            order=1,
            dx=dx_tuple[i],
            accuracy=accuracy,
            kind="central",
            boundary=boundary,
        )
        partials.append(partial)

    # Stack results
    # If dim is None (all dims), result is (ndim, *field.shape)
    # If dim specifies subset, insert gradient components before spatial dims
    if dim is None:
        result = torch.stack(partials, dim=0)
    else:
        # Find where to insert the gradient dimension
        # Insert before the first spatial dimension
        min_spatial_dim = min(spatial_dims)
        result = torch.stack(partials, dim=min_spatial_dim)

    return result


def laplacian(
    field: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
) -> Tensor:
    """Compute Laplacian of a scalar field.

    The Laplacian is the sum of second partial derivatives: nabla^2 f = sum_i d^2f/dx_i^2.

    Parameters
    ----------
    field : Tensor
        Input scalar field with shape (..., *spatial_dims).
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0.
    dim : tuple of int, optional
        Spatial dimensions over which to compute Laplacian. Default uses all dimensions.
    accuracy : int, optional
        Accuracy order of the finite difference approximation. Default is 2.
    boundary : str, optional
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate".

    Returns
    -------
    Tensor
        Laplacian field with the same shape as the input.

    Examples
    --------
    >>> # Laplacian of x^2 + y^2 is 4
    >>> x = torch.linspace(0, 1, 21)
    >>> y = torch.linspace(0, 1, 21)
    >>> X, Y = torch.meshgrid(x, y, indexing="ij")
    >>> f = X**2 + Y**2
    >>> lap = laplacian(f, dx=0.05)  # Should be ~4
    """
    ndim = field.ndim

    # Determine which dimensions to differentiate
    if dim is None:
        spatial_dims = tuple(range(ndim))
    else:
        spatial_dims = tuple(d if d >= 0 else ndim + d for d in dim)

    n_spatial = len(spatial_dims)

    # Handle dx
    if isinstance(dx, (int, float)):
        dx_tuple = (float(dx),) * n_spatial
    else:
        dx_tuple = tuple(float(d) for d in dx)
        if len(dx_tuple) != n_spatial:
            raise ValueError(
                f"dx has {len(dx_tuple)} elements but {n_spatial} spatial dimensions"
            )

    # Sum second derivatives
    result = torch.zeros_like(field)
    for i, d in enumerate(spatial_dims):
        second_deriv = derivative(
            field,
            dim=d,
            order=2,
            dx=dx_tuple[i],
            accuracy=accuracy,
            kind="central",
            boundary=boundary,
        )
        result = result + second_deriv

    return result


def hessian(
    field: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
) -> Tensor:
    """Compute Hessian matrix of a scalar field.

    The Hessian is the matrix of all second partial derivatives:
    H[i,j] = d^2f / (dx_i dx_j).

    Parameters
    ----------
    field : Tensor
        Input scalar field with shape (..., *spatial_dims).
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0.
    dim : tuple of int, optional
        Spatial dimensions over which to compute Hessian. Default uses all dimensions.
    accuracy : int, optional
        Accuracy order of the finite difference approximation. Default is 2.
    boundary : str, optional
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate".

    Returns
    -------
    Tensor
        Hessian field with shape (ndim, ndim, *field.shape) if dim is None,
        or (len(dim), len(dim), *field.shape) if dim is specified.

    Examples
    --------
    >>> # Hessian of 2x^2 + 3xy + 4y^2 is [[4, 3], [3, 8]]
    >>> x = torch.linspace(0, 1, 21)
    >>> y = torch.linspace(0, 1, 21)
    >>> X, Y = torch.meshgrid(x, y, indexing="ij")
    >>> f = 2*X**2 + 3*X*Y + 4*Y**2
    >>> H = hessian(f, dx=0.05)  # Shape: (2, 2, 21, 21)
    """
    ndim = field.ndim

    # Determine which dimensions to differentiate
    if dim is None:
        spatial_dims = tuple(range(ndim))
    else:
        spatial_dims = tuple(d if d >= 0 else ndim + d for d in dim)

    n_spatial = len(spatial_dims)

    # Handle dx
    if isinstance(dx, (int, float)):
        dx_tuple = (float(dx),) * n_spatial
    else:
        dx_tuple = tuple(float(d) for d in dx)
        if len(dx_tuple) != n_spatial:
            raise ValueError(
                f"dx has {len(dx_tuple)} elements but {n_spatial} spatial dimensions"
            )

    # Compute all second partial derivatives
    hessian_components = []

    for i, di in enumerate(spatial_dims):
        row = []
        for j, dj in enumerate(spatial_dims):
            if i == j:
                # Diagonal: d^2f/dx_i^2
                h_ij = derivative(
                    field,
                    dim=di,
                    order=2,
                    dx=dx_tuple[i],
                    accuracy=accuracy,
                    kind="central",
                    boundary=boundary,
                )
            elif j > i:
                # Off-diagonal: d^2f/(dx_i dx_j)
                # Compute as d/dx_j (d/dx_i f)
                df_di = derivative(
                    field,
                    dim=di,
                    order=1,
                    dx=dx_tuple[i],
                    accuracy=accuracy,
                    kind="central",
                    boundary=boundary,
                )
                h_ij = derivative(
                    df_di,
                    dim=dj,
                    order=1,
                    dx=dx_tuple[j],
                    accuracy=accuracy,
                    kind="central",
                    boundary=boundary,
                )
            else:
                # Use symmetry: H[i,j] = H[j,i]
                h_ij = hessian_components[j][i]
            row.append(h_ij)
        hessian_components.append(row)

    # Stack into tensor with shape (n_spatial, n_spatial, *field.shape)
    rows = [torch.stack(row, dim=0) for row in hessian_components]
    result = torch.stack(rows, dim=0)

    return result


def biharmonic(
    field: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
) -> Tensor:
    """Compute biharmonic operator of a scalar field.

    The biharmonic is the Laplacian of the Laplacian: nabla^4 f = nabla^2(nabla^2 f).

    Parameters
    ----------
    field : Tensor
        Input scalar field with shape (..., *spatial_dims).
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0.
    dim : tuple of int, optional
        Spatial dimensions over which to compute biharmonic. Default uses all dimensions.
    accuracy : int, optional
        Accuracy order of the finite difference approximation. Default is 2.
    boundary : str, optional
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate".

    Returns
    -------
    Tensor
        Biharmonic field with the same shape as the input.

    Examples
    --------
    >>> f = torch.randn(20, 20)
    >>> biharm = biharmonic(f, dx=0.1)  # Shape: (20, 20)
    """
    # Compute Laplacian twice
    lap = laplacian(
        field, dx=dx, dim=dim, accuracy=accuracy, boundary=boundary
    )
    return laplacian(lap, dx=dx, dim=dim, accuracy=accuracy, boundary=boundary)
