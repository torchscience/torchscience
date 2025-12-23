from typing import Callable

import torch
from torch import Tensor


def _get_default_tol(dtype: torch.dtype) -> float:
    """Get dtype-aware default tolerance."""
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-3
    elif dtype == torch.float32:
        return 1e-6
    else:  # float64
        return 1e-12


def brent(
    f: Callable[[Tensor], Tensor],
    a: Tensor,
    b: Tensor,
    *,
    xtol: float | None = None,
    ftol: float | None = None,
    maxiter: int = 100,
) -> Tensor:
    """
    Find roots of f(x) = 0 using Brent's method.

    Parameters
    ----------
    f : Callable[[Tensor], Tensor]
        Vectorized function. Takes tensor of shape (N,), returns (N,).
    a, b : Tensor
        Bracket endpoints. Shape (N,). Must satisfy f(a) * f(b) < 0.
    xtol : float, optional
        Tolerance on interval width. Default: dtype-aware.
    ftol : float, optional
        Tolerance on |f(x)|. Default: dtype-aware.
    maxiter : int
        Maximum iterations. Raises RuntimeError if exceeded.

    Returns
    -------
    Tensor
        Roots of shape (N,).
    """
    raise NotImplementedError("brent not yet implemented")
