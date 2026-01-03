"""Vector field differential operators.

High-level functions for computing divergence, curl, and Jacobian on vector fields.
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor

from torchscience.differentiation._scalar_operators import derivative


def divergence(
    vector_field: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
) -> Tensor:
    """Compute divergence of a vector field.

    The divergence is the sum of partial derivatives of each component with respect
    to its corresponding coordinate: div(V) = sum_i dV_i/dx_i.

    Parameters
    ----------
    vector_field : Tensor
        Input vector field with shape (..., ndim, *spatial_dims) where the
        component dimension contains the vector components and spatial_dims
        are the spatial dimensions.
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0.
    dim : tuple of int, optional
        Spatial dimensions over which to compute divergence. Default uses all
        dimensions after the component dimension.
    accuracy : int, optional
        Accuracy order of the finite difference approximation. Default is 2.
    boundary : str, optional
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate".

    Returns
    -------
    Tensor
        Divergence field with shape (..., *spatial_dims).

    Examples
    --------
    >>> # Divergence of (x, y) is 2
    >>> x = torch.linspace(0, 1, 21)
    >>> y = torch.linspace(0, 1, 21)
    >>> X, Y = torch.meshgrid(x, y, indexing="ij")
    >>> V = torch.stack([X, Y], dim=0)  # Shape: (2, 21, 21)
    >>> div = divergence(V, dx=0.05)  # Shape: (21, 21)
    """
    ndim = vector_field.ndim

    # Assume first dimension is the component dimension (by default)
    # The number of components determines the number of spatial dimensions
    n_components = vector_field.shape[0]

    # Determine spatial dimensions
    if dim is None:
        # Use dimensions 1, 2, ..., n_components as spatial dims
        spatial_dims = tuple(range(1, n_components + 1))
    else:
        spatial_dims = tuple(d if d >= 0 else ndim + d for d in dim)

    n_spatial = len(spatial_dims)

    if n_components != n_spatial:
        raise ValueError(
            f"Number of vector components ({n_components}) must match "
            f"number of spatial dimensions ({n_spatial})"
        )

    # Handle dx
    if isinstance(dx, (int, float)):
        dx_tuple = (float(dx),) * n_spatial
    else:
        dx_tuple = tuple(float(d) for d in dx)
        if len(dx_tuple) != n_spatial:
            raise ValueError(
                f"dx has {len(dx_tuple)} elements but {n_spatial} spatial dimensions"
            )

    # Compute divergence: sum of dV_i/dx_i
    result = None
    for i in range(n_components):
        # Get component i: shape (..., *spatial_dims)
        component = vector_field[i]

        # When we index vector_field[i], the component dimension is removed,
        # so spatial dims are shifted down by 1
        spatial_dim_in_component = spatial_dims[i] - 1

        # Compute derivative of component i w.r.t. coordinate i
        partial = derivative(
            component,
            dim=spatial_dim_in_component,
            order=1,
            dx=dx_tuple[i],
            accuracy=accuracy,
            kind="central",
            boundary=boundary,
        )

        if result is None:
            result = partial
        else:
            result = result + partial

    return result


def curl(
    vector_field: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
) -> Tensor:
    """Compute curl of a 3D vector field.

    The curl is the vector field defined by:
    curl(V) = (dVz/dy - dVy/dz, dVx/dz - dVz/dx, dVy/dx - dVx/dy)

    Parameters
    ----------
    vector_field : Tensor
        Input 3D vector field with shape (..., 3, nx, ny, nz).
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0.
    dim : tuple of int, optional
        Spatial dimensions over which to compute curl. Must have length 3.
        Default uses dimensions 1, 2, 3 (after the component dimension).
    accuracy : int, optional
        Accuracy order of the finite difference approximation. Default is 2.
    boundary : str, optional
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate".

    Returns
    -------
    Tensor
        Curl field with shape (..., 3, nx, ny, nz).

    Examples
    --------
    >>> # Curl of gradient is zero
    >>> V = torch.randn(3, 10, 10, 10)
    >>> c = curl(V, dx=0.1)  # Shape: (3, 10, 10, 10)

    Raises
    ------
    ValueError
        If the vector field is not 3D (does not have exactly 3 components).
    """
    ndim = vector_field.ndim
    n_components = vector_field.shape[0]

    if n_components != 3:
        raise ValueError(
            f"Curl requires 3D vector field, got {n_components} components"
        )

    # Determine spatial dimensions
    if dim is None:
        spatial_dims = (1, 2, 3)
    else:
        spatial_dims = tuple(d if d >= 0 else ndim + d for d in dim)

    if len(spatial_dims) != 3:
        raise ValueError(
            f"Curl requires exactly 3 spatial dimensions, got {len(spatial_dims)}"
        )

    # Handle dx
    if isinstance(dx, (int, float)):
        dx_tuple = (float(dx), float(dx), float(dx))
    else:
        dx_tuple = tuple(float(d) for d in dx)
        if len(dx_tuple) != 3:
            raise ValueError(
                f"dx must have 3 elements for 3D curl, got {len(dx_tuple)}"
            )

    # Extract components: Vx, Vy, Vz
    Vx = vector_field[0]  # Shape: (nx, ny, nz)
    Vy = vector_field[1]
    Vz = vector_field[2]

    # Spatial dims in components (shifted down by 1 since component dim is removed)
    dim_x = spatial_dims[0] - 1  # x dimension in component tensor
    dim_y = spatial_dims[1] - 1  # y dimension in component tensor
    dim_z = spatial_dims[2] - 1  # z dimension in component tensor

    dx_val, dy_val, dz_val = dx_tuple

    # Compute curl components:
    # curl_x = dVz/dy - dVy/dz
    dVz_dy = derivative(
        Vz, dim=dim_y, order=1, dx=dy_val, accuracy=accuracy, boundary=boundary
    )
    dVy_dz = derivative(
        Vy, dim=dim_z, order=1, dx=dz_val, accuracy=accuracy, boundary=boundary
    )
    curl_x = dVz_dy - dVy_dz

    # curl_y = dVx/dz - dVz/dx
    dVx_dz = derivative(
        Vx, dim=dim_z, order=1, dx=dz_val, accuracy=accuracy, boundary=boundary
    )
    dVz_dx = derivative(
        Vz, dim=dim_x, order=1, dx=dx_val, accuracy=accuracy, boundary=boundary
    )
    curl_y = dVx_dz - dVz_dx

    # curl_z = dVy/dx - dVx/dy
    dVy_dx = derivative(
        Vy, dim=dim_x, order=1, dx=dx_val, accuracy=accuracy, boundary=boundary
    )
    dVx_dy = derivative(
        Vx, dim=dim_y, order=1, dx=dy_val, accuracy=accuracy, boundary=boundary
    )
    curl_z = dVy_dx - dVx_dy

    return torch.stack([curl_x, curl_y, curl_z], dim=0)


def jacobian(
    vector_field: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
) -> Tensor:
    """Compute Jacobian matrix of a vector field.

    The Jacobian is the matrix of all partial derivatives:
    J[i, j] = dV_i / dx_j.

    Parameters
    ----------
    vector_field : Tensor
        Input vector field with shape (..., m, *spatial_dims) where m is the
        number of vector components.
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0.
    dim : tuple of int, optional
        Spatial dimensions over which to compute Jacobian. Default uses dimensions
        1, 2, ..., n after the component dimension.
    accuracy : int, optional
        Accuracy order of the finite difference approximation. Default is 2.
    boundary : str, optional
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate".

    Returns
    -------
    Tensor
        Jacobian field with shape (..., m, ndim, *spatial_dims) where m is the
        number of components and ndim is the number of spatial dimensions.

    Examples
    --------
    >>> # Jacobian of (2x + 3y, 4x + 5y) is [[2, 3], [4, 5]]
    >>> x = torch.linspace(0, 1, 21)
    >>> y = torch.linspace(0, 1, 21)
    >>> X, Y = torch.meshgrid(x, y, indexing="ij")
    >>> V = torch.stack([2*X + 3*Y, 4*X + 5*Y], dim=0)  # Shape: (2, 21, 21)
    >>> J = jacobian(V, dx=0.05)  # Shape: (2, 2, 21, 21)
    """
    ndim = vector_field.ndim
    n_components = vector_field.shape[0]

    # Determine spatial dimensions
    if dim is None:
        # Infer spatial dimensions based on tensor shape
        # Assume first dimension is components, rest are spatial
        n_spatial = ndim - 1
        spatial_dims = tuple(range(1, ndim))
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

    # Compute Jacobian: J[i, j] = dV_i / dx_j
    jacobian_rows = []

    for i in range(n_components):
        # Get component i: shape (*spatial_dims)
        component = vector_field[i]

        row = []
        for j, spatial_dim in enumerate(spatial_dims):
            # When we index vector_field[i], the component dimension is removed,
            # so spatial dims are shifted down by 1
            dim_in_component = spatial_dim - 1

            # Compute dV_i / dx_j
            partial = derivative(
                component,
                dim=dim_in_component,
                order=1,
                dx=dx_tuple[j],
                accuracy=accuracy,
                kind="central",
                boundary=boundary,
            )
            row.append(partial)

        # Stack row: shape (n_spatial, *spatial_dims)
        jacobian_rows.append(torch.stack(row, dim=0))

    # Stack all rows: shape (n_components, n_spatial, *spatial_dims)
    result = torch.stack(jacobian_rows, dim=0)

    return result
