"""Chebyshev Vandermonde matrix."""

from __future__ import annotations

import torch
from torch import Tensor


def chebyshev_t_vandermonde(x: Tensor, degree: int) -> Tensor:
    """Generate Chebyshev Vandermonde matrix.

    V[i, j] = T_j(x[i])

    Parameters
    ----------
    x : Tensor
        Evaluation points, shape (n,).
    degree : int
        Maximum degree (matrix has degree+1 columns).

    Returns
    -------
    Tensor
        Vandermonde matrix, shape (n, degree+1).

    Notes
    -----
    Built using the Chebyshev recurrence:
        T_0(x) = 1
        T_1(x) = x
        T_{n+1}(x) = 2*x*T_n(x) - T_{n-1}(x)

    Examples
    --------
    >>> x = torch.tensor([0.0, 0.5, 1.0])
    >>> chebyshev_t_vandermonde(x, degree=2)
    tensor([[ 1.0000,  0.0000, -1.0000],
            [ 1.0000,  0.5000, -0.5000],
            [ 1.0000,  1.0000,  1.0000]])
    """
    n = x.shape[0]
    V = torch.zeros(n, degree + 1, dtype=x.dtype, device=x.device)

    # T_0 = 1
    V[:, 0] = 1.0

    if degree >= 1:
        # T_1 = x
        V[:, 1] = x

    # Recurrence: T_{k+1} = 2*x*T_k - T_{k-1}
    for k in range(1, degree):
        V[:, k + 1] = 2.0 * x * V[:, k] - V[:, k - 1]

    return V
