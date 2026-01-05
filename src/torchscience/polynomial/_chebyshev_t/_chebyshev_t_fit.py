"""Least-squares Chebyshev series fit."""

from __future__ import annotations

import torch
from torch import Tensor

from ._chebyshev_t import ChebyshevT
from ._chebyshev_t_vandermonde import chebyshev_t_vandermonde


def chebyshev_t_fit(x: Tensor, y: Tensor, degree: int) -> ChebyshevT:
    """Fit Chebyshev series to data using least squares.

    Parameters
    ----------
    x : Tensor
        Sample points, shape (n,).
    y : Tensor
        Sample values, shape (n,).
    degree : int
        Degree of fitting polynomial.

    Returns
    -------
    ChebyshevT
        Fitted Chebyshev series.

    Notes
    -----
    Uses the normal equations via torch.linalg.lstsq for numerical stability.

    Examples
    --------
    >>> x = torch.linspace(-1, 1, 10)
    >>> y = x**2
    >>> c = chebyshev_t_fit(x, y, degree=2)
    """
    # Build Vandermonde matrix
    V = chebyshev_t_vandermonde(x, degree)

    # Solve least squares: V @ coeffs = y
    # Use lstsq for numerical stability
    result = torch.linalg.lstsq(V, y.unsqueeze(-1))
    coeffs = result.solution.squeeze(-1)

    return ChebyshevT(coeffs=coeffs)
