from __future__ import annotations

from torch import Tensor

from ._polynomial import Polynomial


def polynomial_scale(p: Polynomial, c: Tensor) -> Polynomial:
    """Multiply polynomial by scalar(s).

    Parameters
    ----------
    p : Polynomial
        Polynomial to scale.
    c : Tensor
        Scalar(s), broadcasts with batch dimensions.

    Returns
    -------
    Polynomial
        Scaled polynomial c * p.
    """
    # Ensure c can broadcast with coeffs
    if c.dim() == 0:
        return Polynomial(coeffs=p.coeffs * c)
    else:
        # c broadcasts with batch dimensions, not coefficient dimension
        return Polynomial(coeffs=p.coeffs * c.unsqueeze(-1))
