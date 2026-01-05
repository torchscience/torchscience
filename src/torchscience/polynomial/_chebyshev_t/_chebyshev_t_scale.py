"""Scale a Chebyshev series by a scalar."""

from __future__ import annotations

from torch import Tensor

from ._chebyshev_t import ChebyshevT


def chebyshev_t_scale(a: ChebyshevT, scalar: Tensor) -> ChebyshevT:
    """Scale a Chebyshev series by a scalar.

    Parameters
    ----------
    a : ChebyshevT
        Series to scale.
    scalar : Tensor
        Scalar multiplier.

    Returns
    -------
    ChebyshevT
        Scaled series scalar * a.

    Examples
    --------
    >>> a = chebyshev_t(torch.tensor([1.0, 2.0, 3.0]))
    >>> b = chebyshev_t_scale(a, torch.tensor(2.0))
    >>> b.coeffs
    tensor([2., 4., 6.])
    """
    return ChebyshevT(coeffs=a.coeffs * scalar)
