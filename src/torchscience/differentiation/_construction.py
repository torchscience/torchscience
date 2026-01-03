"""Stencil construction using Fornberg's algorithm."""

from __future__ import annotations

from collections import defaultdict
from typing import Tuple, Union

import torch
from torch import Tensor

from torchscience.differentiation._exceptions import StencilError
from torchscience.differentiation._stencil import FiniteDifferenceStencil


def fornberg_weights(
    derivative_order: int,
    offsets: Tensor,
    x0: float = 0.0,
) -> Tensor:
    """Compute finite difference weights using Fornberg's algorithm.

    This implements the algorithm from:
    Fornberg, B. (1988). "Generation of Finite Difference Formulas on
    Arbitrarily Spaced Grids". Mathematics of Computation, 51(184), 699-706.

    Parameters
    ----------
    derivative_order : int
        Order of derivative (1 for first derivative, 2 for second, etc.).
    offsets : Tensor
        Grid point offsets, shape (n,). Can be non-uniform spacing.
    x0 : float
        Point at which to evaluate the derivative.

    Returns
    -------
    Tensor
        Weights for each grid point, shape (n,).
    """
    n = len(offsets)
    m = derivative_order

    if n <= m:
        raise StencilError(
            f"Need at least {m + 1} points for derivative order {m}, got {n}"
        )

    x = offsets.float() - x0  # Grid points relative to evaluation point
    dtype = offsets.dtype if offsets.is_floating_point() else torch.float64

    # Initialize coefficient table
    # c[j, k] = coefficient for point j when computing k-th derivative
    c = torch.zeros(n, m + 1, dtype=torch.float64, device=offsets.device)

    c[0, 0] = 1.0
    c1 = 1.0

    for j in range(1, n):
        c2 = 1.0
        for k in range(j):
            c3 = x[j] - x[k]
            c2 = c2 * c3

            for s in range(min(j, m), 0, -1):
                c[j, s] = c1 * (s * c[j - 1, s - 1] - x[k] * c[j - 1, s]) / c2

            c[j, 0] = -c1 * x[k] * c[j - 1, 0] / c2

            for s in range(min(j, m), 0, -1):
                c[k, s] = (x[j] * c[k, s] - s * c[k, s - 1]) / c3

            c[k, 0] = x[j] * c[k, 0] / c3

        c1 = c2

    # Extract weights for the requested derivative order
    weights = c[:, m].to(dtype)
    return weights


def finite_difference_stencil(
    derivative: Union[int, Tuple[int, ...]],
    accuracy: int = 2,
    kind: str = "central",
    offsets: Tensor | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> FiniteDifferenceStencil:
    """Generate finite difference stencil for arbitrary derivative.

    Uses Fornberg's algorithm to compute optimal coefficients for the
    given grid points and derivative order.

    Parameters
    ----------
    derivative : int or tuple of int
        Derivative order. For 1D, can be an int. For n-D, use tuple like
        (2, 0) for d^2/dx^2 or (1, 1) for mixed partial d^2/dxdy.
    accuracy : int
        Accuracy order of the approximation. Error is O(dx^accuracy).
        Must be positive. Default is 2 (second-order accuracy).
    kind : str
        Stencil type: "central", "forward", or "backward".
        - "central": symmetric around the point (most accurate per point)
        - "forward": uses points at and ahead of the point
        - "backward": uses points at and behind the point
    offsets : Tensor, optional
        Custom grid point offsets. If provided, overrides `kind`.
        Shape should be (n_points,) for 1D or (n_points, ndim) for n-D.
    dtype : torch.dtype, optional
        Output dtype. Default is torch.float64.
    device : torch.device, optional
        Output device. Default is CPU.

    Returns
    -------
    FiniteDifferenceStencil
        Stencil with offsets and coefficients.

    Raises
    ------
    ValueError
        If derivative <= 0, accuracy <= 0, or invalid kind.

    Examples
    --------
    >>> stencil = finite_difference_stencil(derivative=1, accuracy=2)
    >>> stencil.coeffs  # Central first derivative: [-0.5, 0, 0.5]

    >>> stencil = finite_difference_stencil(derivative=2, accuracy=4)
    >>> stencil.coeffs  # Central second derivative, 4th order accuracy

    >>> stencil = finite_difference_stencil(derivative=(1, 1), accuracy=2)
    >>> stencil.coeffs  # Mixed partial d^2/dxdy
    """
    # Normalize derivative to tuple
    if isinstance(derivative, int):
        derivative_tuple = (derivative,)
    else:
        derivative_tuple = tuple(derivative)

    ndim = len(derivative_tuple)

    # Validation
    if any(d < 0 for d in derivative_tuple):
        raise ValueError("derivative orders must be non-negative")
    if sum(derivative_tuple) == 0:
        raise ValueError("total derivative order must be positive")
    if accuracy <= 0:
        raise ValueError("accuracy must be positive")
    if kind not in ("central", "forward", "backward"):
        raise ValueError(
            f"kind must be 'central', 'forward', or 'backward', got '{kind}'"
        )

    if dtype is None:
        dtype = torch.float64
    if device is None:
        device = torch.device("cpu")

    # Generate offsets if not provided
    if offsets is not None:
        # Custom offsets provided
        if offsets.dim() == 1:
            offsets = offsets.unsqueeze(-1)  # (n,) -> (n, 1)
        custom_offsets = offsets.to(device=device)
    else:
        # Generate standard offsets based on kind and accuracy
        custom_offsets = None

    if ndim == 1:
        # 1D case
        deriv_order = derivative_tuple[0]

        if custom_offsets is not None:
            grid_offsets = custom_offsets.squeeze(-1)
        else:
            grid_offsets = _generate_1d_offsets(
                deriv_order, accuracy, kind, device
            )

        # Compute weights using Fornberg's algorithm
        weights = fornberg_weights(deriv_order, grid_offsets)

        return FiniteDifferenceStencil(
            offsets=grid_offsets.unsqueeze(-1).long(),
            coeffs=weights.to(dtype),
            derivative=derivative_tuple,
            accuracy=accuracy,
        )
    else:
        # Multi-dimensional case: tensor product of 1D stencils
        return _generate_nd_stencil(
            derivative_tuple, accuracy, kind, dtype, device
        )


def _generate_1d_offsets(
    deriv_order: int,
    accuracy: int,
    kind: str,
    device: torch.device,
) -> Tensor:
    """Generate 1D offset grid for given parameters.

    For central differences, the minimum number of points to achieve
    p-th order accuracy for a d-th derivative is:
        n = d + p - 1 + ((d + p) mod 2)

    This ensures we have an odd number of points (symmetric around center)
    and enough points to cancel the required Taylor series terms.
    """
    if kind == "central":
        # Minimal central stencil: need odd number of points >= deriv_order + 1
        # that achieves the requested accuracy
        n_points = deriv_order + accuracy - 1 + ((deriv_order + accuracy) % 2)
        half = n_points // 2
        offsets = torch.arange(-half, half + 1, device=device)
    elif kind == "forward":
        # Forward differences: need deriv_order + accuracy points starting at 0
        n_points = deriv_order + accuracy
        offsets = torch.arange(0, n_points, device=device)
    else:  # backward
        # Backward differences: need deriv_order + accuracy points ending at 0
        n_points = deriv_order + accuracy
        offsets = torch.arange(-(n_points - 1), 1, device=device)

    return offsets.float()


def _generate_nd_stencil(
    derivative: Tuple[int, ...],
    accuracy: int,
    kind: str,
    dtype: torch.dtype,
    device: torch.device,
) -> FiniteDifferenceStencil:
    """Generate n-dimensional stencil via tensor product."""
    ndim = len(derivative)

    # Generate 1D stencils for each dimension
    stencils_1d = []
    for d in range(ndim):
        deriv_order = derivative[d]
        if deriv_order == 0:
            # Identity in this dimension: single point at offset 0
            stencils_1d.append(
                (
                    torch.tensor([0], device=device),
                    torch.tensor([1.0], device=device),
                )
            )
        else:
            offsets_1d = _generate_1d_offsets(
                deriv_order, accuracy, kind, device
            )
            weights_1d = fornberg_weights(deriv_order, offsets_1d)
            stencils_1d.append((offsets_1d.long(), weights_1d))

    # Compute tensor product
    all_offsets = []
    all_coeffs = []

    # Use recursive Cartesian product
    def cartesian_product(
        dim: int, current_offset: list, current_coeff: float
    ):
        if dim == ndim:
            all_offsets.append(current_offset.copy())
            all_coeffs.append(current_coeff)
            return

        offsets_d, weights_d = stencils_1d[dim]
        for i in range(len(offsets_d)):
            current_offset.append(offsets_d[i].item())
            cartesian_product(
                dim + 1, current_offset, current_coeff * weights_d[i].item()
            )
            current_offset.pop()

    cartesian_product(0, [], 1.0)

    offsets_tensor = torch.tensor(
        all_offsets, dtype=torch.int64, device=device
    )
    coeffs_tensor = torch.tensor(all_coeffs, dtype=dtype, device=device)

    # Filter out near-zero coefficients for sparse representation
    nonzero_mask = coeffs_tensor.abs() > 1e-14
    offsets_tensor = offsets_tensor[nonzero_mask]
    coeffs_tensor = coeffs_tensor[nonzero_mask]

    return FiniteDifferenceStencil(
        offsets=offsets_tensor,
        coeffs=coeffs_tensor,
        derivative=derivative,
        accuracy=accuracy,
    )


def laplacian_stencil(
    ndim: int,
    accuracy: int = 2,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> FiniteDifferenceStencil:
    """Generate n-dimensional Laplacian stencil.

    The Laplacian is the sum of second derivatives in each dimension:
    nabla^2 f = d^2f/dx^2 + d^2f/dy^2 + ...

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions.
    accuracy : int
        Accuracy order of the approximation. Default is 2.
    dtype : torch.dtype, optional
        Output dtype. Default is torch.float64.
    device : torch.device, optional
        Output device. Default is CPU.

    Returns
    -------
    FiniteDifferenceStencil
        Laplacian stencil with combined offsets and coefficients.
        For 2D with accuracy=2, this is the 5-point stencil.
        For 3D with accuracy=2, this is the 7-point stencil.

    Examples
    --------
    >>> stencil = laplacian_stencil(ndim=2, accuracy=2)
    >>> stencil.n_points  # 5-point stencil
    5
    """
    if ndim < 1:
        raise ValueError("ndim must be at least 1")
    if accuracy <= 0:
        raise ValueError("accuracy must be positive")

    if dtype is None:
        dtype = torch.float64
    if device is None:
        device = torch.device("cpu")

    # Generate 1D second derivative stencil
    stencil_1d = finite_difference_stencil(
        derivative=2,
        accuracy=accuracy,
        kind="central",
        dtype=dtype,
        device=device,
    )

    # Combine 1D stencils embedded in n-D
    # Each dimension contributes offsets along its axis
    offset_to_coeff: defaultdict[tuple, float] = defaultdict(float)

    for dim in range(ndim):
        for i in range(stencil_1d.n_points):
            # Create n-D offset with the 1D offset in position dim
            offset = [0] * ndim
            offset[dim] = stencil_1d.offsets[i, 0].item()
            offset_tuple = tuple(offset)
            offset_to_coeff[offset_tuple] += stencil_1d.coeffs[i].item()

    # Convert to tensors, filtering near-zero coefficients
    offsets_list = []
    coeffs_list = []
    for offset, coeff in offset_to_coeff.items():
        if abs(coeff) > 1e-14:
            offsets_list.append(list(offset))
            coeffs_list.append(coeff)

    offsets_tensor = torch.tensor(
        offsets_list, dtype=torch.int64, device=device
    )
    coeffs_tensor = torch.tensor(coeffs_list, dtype=dtype, device=device)

    # The derivative tuple for Laplacian is (2, 2, ...) representing
    # sum of second derivatives (though this is a sum, not a product)
    derivative_tuple = tuple([2] * ndim)

    return FiniteDifferenceStencil(
        offsets=offsets_tensor,
        coeffs=coeffs_tensor,
        derivative=derivative_tuple,
        accuracy=accuracy,
    )


def gradient_stencils(
    ndim: int,
    accuracy: int = 2,
    kind: str = "central",
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tuple[FiniteDifferenceStencil, ...]:
    """Generate tuple of gradient stencils, one per dimension.

    The gradient is a vector of first partial derivatives:
    grad f = (df/dx, df/dy, ...)

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions.
    accuracy : int
        Accuracy order of the approximation. Default is 2.
    kind : str
        Stencil type: "central", "forward", or "backward".
    dtype : torch.dtype, optional
        Output dtype. Default is torch.float64.
    device : torch.device, optional
        Output device. Default is CPU.

    Returns
    -------
    Tuple[FiniteDifferenceStencil, ...]
        Tuple of n stencils, one for each dimension.
        stencils[i].derivative has 1 in position i and 0 elsewhere.

    Examples
    --------
    >>> stencils = gradient_stencils(ndim=2, accuracy=2)
    >>> len(stencils)
    2
    >>> stencils[0].derivative
    (1, 0)
    >>> stencils[1].derivative
    (0, 1)
    """
    if ndim < 1:
        raise ValueError("ndim must be at least 1")
    if accuracy <= 0:
        raise ValueError("accuracy must be positive")

    stencils = []
    for dim in range(ndim):
        # Create derivative tuple with 1 in position dim
        derivative = tuple(1 if d == dim else 0 for d in range(ndim))
        stencil = finite_difference_stencil(
            derivative=derivative,
            accuracy=accuracy,
            kind=kind,
            dtype=dtype,
            device=device,
        )
        stencils.append(stencil)

    return tuple(stencils)


def biharmonic_stencil(
    ndim: int,
    accuracy: int = 2,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> FiniteDifferenceStencil:
    """Generate biharmonic (nabla^4) stencil.

    The biharmonic operator is the Laplacian applied twice:
    nabla^4 f = nabla^2(nabla^2 f)

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions.
    accuracy : int
        Accuracy order of the approximation. Default is 2.
    dtype : torch.dtype, optional
        Output dtype. Default is torch.float64.
    device : torch.device, optional
        Output device. Default is CPU.

    Returns
    -------
    FiniteDifferenceStencil
        Biharmonic stencil computed by self-convolving the Laplacian.
        For 2D with accuracy=2, this is a 13-point stencil.

    Examples
    --------
    >>> stencil = biharmonic_stencil(ndim=2, accuracy=2)
    >>> stencil.n_points  # 13-point stencil
    13
    """
    if ndim < 1:
        raise ValueError("ndim must be at least 1")
    if accuracy <= 0:
        raise ValueError("accuracy must be positive")

    if dtype is None:
        dtype = torch.float64
    if device is None:
        device = torch.device("cpu")

    # Get Laplacian stencil
    lap = laplacian_stencil(
        ndim=ndim, accuracy=accuracy, dtype=dtype, device=device
    )

    # Self-convolve: for each pair of offsets, add them and multiply coefficients
    offset_to_coeff: defaultdict[tuple, float] = defaultdict(float)

    for i in range(lap.n_points):
        for j in range(lap.n_points):
            # Sum offsets
            offset_i = lap.offsets[i].tolist()
            offset_j = lap.offsets[j].tolist()
            combined_offset = tuple(a + b for a, b in zip(offset_i, offset_j))

            # Multiply coefficients
            combined_coeff = lap.coeffs[i].item() * lap.coeffs[j].item()
            offset_to_coeff[combined_offset] += combined_coeff

    # Convert to tensors, filtering near-zero coefficients
    offsets_list = []
    coeffs_list = []
    for offset, coeff in offset_to_coeff.items():
        if abs(coeff) > 1e-14:
            offsets_list.append(list(offset))
            coeffs_list.append(coeff)

    offsets_tensor = torch.tensor(
        offsets_list, dtype=torch.int64, device=device
    )
    coeffs_tensor = torch.tensor(coeffs_list, dtype=dtype, device=device)

    # The derivative tuple for biharmonic is (4, 4, ...) representing
    # the fourth-order operator in each dimension
    derivative_tuple = tuple([4] * ndim)

    return FiniteDifferenceStencil(
        offsets=offsets_tensor,
        coeffs=coeffs_tensor,
        derivative=derivative_tuple,
        accuracy=accuracy,
    )
