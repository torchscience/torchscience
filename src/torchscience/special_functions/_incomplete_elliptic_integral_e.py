import torch
from torch import Tensor

import torchscience

__all__ = ["incomplete_elliptic_integral_e"]


def incomplete_elliptic_integral_e(phi: Tensor, k: Tensor) -> Tensor:
    r"""
    Incomplete elliptic integral of the second kind :math:`E(\phi, k)`.

    .. math::
        E(\phi, k) = \int_0^{\phi} \sqrt{1 - k^2 \sin^2 t} \, dt

    This integral computes the arc length of an ellipse.

    Args:
        phi (Tensor): The amplitude (upper limit of integration, in radians).
        k (Tensor): The elliptic modulus (must satisfy |k| ≤ 1).

    Returns:
        Tensor: The value of the incomplete elliptic integral of the second kind.
    """
    return torchscience.ops.torchscience._incomplete_elliptic_integral_e(phi, k)
