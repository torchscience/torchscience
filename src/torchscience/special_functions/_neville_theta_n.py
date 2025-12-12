import torch
from torch import Tensor

import torchscience

__all__ = ["neville_theta_n"]


def neville_theta_n(k: Tensor, u: Tensor) -> Tensor:
    r"""
    Neville theta function :math:`\theta_n(k, u)`.

    The Neville theta functions are related to the Jacobi theta functions
    and are used in the theory of elliptic functions.

    Args:
        k (Tensor): Input tensor for the elliptic modulus k (0 ≤ k ≤ 1).
        u (Tensor): Input tensor for the argument u.

    Returns:
        Tensor: The value of the Neville theta function θn(k, u).
    """
    return torchscience.ops.torchscience._neville_theta_n(k, u)
