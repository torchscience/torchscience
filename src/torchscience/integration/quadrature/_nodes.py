"""Node and weight computation for quadrature rules."""

from typing import Optional, Tuple

import torch
from torch import Tensor


def gauss_legendre_nodes_weights(
    n: int,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Compute Gauss-Legendre nodes and weights on [-1, 1].

    Uses the Golub-Welsch algorithm (eigenvalues of symmetric tridiagonal matrix).

    Parameters
    ----------
    n : int
        Number of quadrature points.
    dtype : torch.dtype
        Data type for output tensors.
    device : torch.device, optional
        Device for output tensors.

    Returns
    -------
    nodes : Tensor
        Quadrature nodes, shape (n,), sorted ascending.
    weights : Tensor
        Quadrature weights, shape (n,).

    Raises
    ------
    ValueError
        If n < 1.

    Notes
    -----
    Gauss-Legendre quadrature is exact for polynomials of degree <= 2n-1.

    The algorithm constructs the symmetric tridiagonal Jacobi matrix for
    Legendre polynomials and computes its eigenvalues (nodes) and
    eigenvectors (used to compute weights).

    References
    ----------
    Golub, G. H., & Welsch, J. H. (1969). Calculation of Gauss quadrature rules.
    Mathematics of Computation, 23(106), 221-230.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")

    if n == 1:
        return (
            torch.tensor([0.0], dtype=dtype, device=device),
            torch.tensor([2.0], dtype=dtype, device=device),
        )

    # Build symmetric tridiagonal Jacobi matrix for Legendre polynomials
    # For Legendre: diagonal = 0, off-diagonal[k] = k / sqrt(4k^2 - 1)
    k = torch.arange(1, n, dtype=dtype, device=device)
    off_diag = k / torch.sqrt(4 * k**2 - 1)

    # Construct tridiagonal matrix
    T = torch.diag(off_diag, diagonal=1) + torch.diag(off_diag, diagonal=-1)

    # Eigenvalues are nodes, first components of eigenvectors give weights
    eigenvalues, eigenvectors = torch.linalg.eigh(T)

    nodes = eigenvalues
    weights = 2 * eigenvectors[0, :] ** 2

    # Sort by nodes (should already be sorted from eigh, but ensure)
    sorted_idx = torch.argsort(nodes)
    nodes = nodes[sorted_idx]
    weights = weights[sorted_idx]

    return nodes, weights
