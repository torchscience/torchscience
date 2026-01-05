"""Negate a Chebyshev series."""

from __future__ import annotations

from ._chebyshev_t import ChebyshevT


def chebyshev_t_negate(a: ChebyshevT) -> ChebyshevT:
    """Negate a Chebyshev series.

    Parameters
    ----------
    a : ChebyshevT
        Series to negate.

    Returns
    -------
    ChebyshevT
        Negated series -a.

    Examples
    --------
    >>> a = chebyshev_t(torch.tensor([1.0, -2.0, 3.0]))
    >>> b = chebyshev_t_negate(a)
    >>> b.coeffs
    tensor([-1.,  2., -3.])
    """
    return ChebyshevT(coeffs=-a.coeffs)
