"""Add two Chebyshev series."""

from __future__ import annotations

import torch

from ._chebyshev_t import ChebyshevT


def chebyshev_t_add(a: ChebyshevT, b: ChebyshevT) -> ChebyshevT:
    """Add two Chebyshev series.

    Parameters
    ----------
    a : ChebyshevT
        First series.
    b : ChebyshevT
        Second series.

    Returns
    -------
    ChebyshevT
        Sum a + b.

    Notes
    -----
    If the series have different degrees, the shorter one is zero-padded.

    Examples
    --------
    >>> a = chebyshev_t(torch.tensor([1.0, 2.0]))
    >>> b = chebyshev_t(torch.tensor([3.0, 4.0, 5.0]))
    >>> c = chebyshev_t_add(a, b)
    >>> c.coeffs
    tensor([4., 6., 5.])
    """
    a_coeffs = a.coeffs
    b_coeffs = b.coeffs

    n_a = a_coeffs.shape[-1]
    n_b = b_coeffs.shape[-1]

    if n_a == n_b:
        return ChebyshevT(coeffs=a_coeffs + b_coeffs)

    # Zero-pad the shorter series
    if n_a < n_b:
        pad_shape = list(a_coeffs.shape)
        pad_shape[-1] = n_b - n_a
        padding = torch.zeros(
            pad_shape, dtype=a_coeffs.dtype, device=a_coeffs.device
        )
        a_coeffs = torch.cat([a_coeffs, padding], dim=-1)
    else:
        pad_shape = list(b_coeffs.shape)
        pad_shape[-1] = n_a - n_b
        padding = torch.zeros(
            pad_shape, dtype=b_coeffs.dtype, device=b_coeffs.device
        )
        b_coeffs = torch.cat([b_coeffs, padding], dim=-1)

    return ChebyshevT(coeffs=a_coeffs + b_coeffs)
