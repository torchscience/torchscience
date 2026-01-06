"""Trim trailing zero coefficients from Chebyshev series."""

from __future__ import annotations

import torch

from ._chebyshev_t import ChebyshevT


def chebyshev_t_trim(c: ChebyshevT, tol: float = 0.0) -> ChebyshevT:
    """Remove trailing coefficients smaller than tolerance.

    Parameters
    ----------
    c : ChebyshevT
        Chebyshev series.
    tol : float, optional
        Tolerance for "small" coefficients. Default is 0.0.

    Returns
    -------
    ChebyshevT
        Trimmed series with at least one coefficient.

    Examples
    --------
    >>> c = chebyshev_t(torch.tensor([1.0, 2.0, 0.0, 0.0]))
    >>> chebyshev_t_trim(c).coeffs
    tensor([1., 2.])
    """
    coeffs = c.coeffs
    n = coeffs.shape[-1]

    # Find last coefficient larger than tolerance
    last_nonzero = 0
    for i in range(n - 1, -1, -1):
        if torch.abs(coeffs[..., i]).max() > tol:
            last_nonzero = i
            break

    # Keep at least one coefficient
    last_nonzero = max(last_nonzero, 0)

    return ChebyshevT(coeffs=coeffs[..., : last_nonzero + 1].clone())
