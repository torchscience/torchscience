"""Construct Chebyshev series from roots."""

from __future__ import annotations

import torch
from torch import Tensor

from ._chebyshev_t import ChebyshevT
from ._chebyshev_t_multiply import chebyshev_t_multiply


def chebyshev_t_from_roots(roots: Tensor) -> ChebyshevT:
    """Construct monic Chebyshev series from its roots.

    The resulting series is (x - r_0)(x - r_1)...(x - r_{n-1}).

    Parameters
    ----------
    roots : Tensor
        Roots of the polynomial, shape (n,).

    Returns
    -------
    ChebyshevT
        Monic Chebyshev series with the given roots.

    Notes
    -----
    Builds the product of linear factors (x - r_k) in Chebyshev form.
    Since x = T_1 and constants are T_0, (x - r) = -r*T_0 + T_1.

    Examples
    --------
    >>> roots = torch.tensor([0.5, -0.5])
    >>> c = chebyshev_t_from_roots(roots)
    """
    n = roots.shape[0]

    if n == 0:
        # Empty roots -> constant 1
        return ChebyshevT(
            coeffs=torch.ones(1, dtype=roots.dtype, device=roots.device)
        )

    # Start with (x - r_0) = -r_0 * T_0 + T_1
    result = ChebyshevT(
        coeffs=torch.tensor(
            [-roots[0], 1.0], dtype=roots.dtype, device=roots.device
        )
    )

    # Multiply by each subsequent (x - r_k)
    for k in range(1, n):
        factor = ChebyshevT(
            coeffs=torch.tensor(
                [-roots[k], 1.0], dtype=roots.dtype, device=roots.device
            )
        )
        result = chebyshev_t_multiply(result, factor)

    return result
