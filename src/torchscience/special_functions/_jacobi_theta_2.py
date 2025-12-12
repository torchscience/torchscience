import torch
from torch import Tensor

import torchscience

__all__ = ["jacobi_theta_2"]


def jacobi_theta_2(z: Tensor, q: Tensor) -> Tensor:
    r"""
    Jacobi theta function :math:`\theta_2(z, q)`.

    .. math::
        \theta_2(z, q) = 2 \sum_{n=0}^{\infty} q^{(n+1/2)^2} \cos((2n+1)z)

    Args:
        z (Tensor): Input tensor for the argument z.
        q (Tensor): Input tensor for the nome q (must satisfy 0 ≤ |q| < 1).

    Returns:
        Tensor: The value of the Jacobi theta function θ₂(z, q).
    """
    return torchscience.ops.torchscience._jacobi_theta_2(z, q)
