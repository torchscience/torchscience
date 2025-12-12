import torch
from torch import Tensor

import torchscience

__all__ = ["jacobi_theta_3"]


def jacobi_theta_3(z: Tensor, q: Tensor) -> Tensor:
    r"""
    Jacobi theta function :math:`\theta_3(z, q)`.

    .. math::
        \theta_3(z, q) = 1 + 2 \sum_{n=1}^{\infty} q^{n^2} \cos(2nz)

    Args:
        z (Tensor): Input tensor for the argument z.
        q (Tensor): Input tensor for the nome q (must satisfy 0 ≤ |q| < 1).

    Returns:
        Tensor: The value of the Jacobi theta function θ₃(z, q).
    """
    return torchscience.ops.torchscience._jacobi_theta_3(z, q)
