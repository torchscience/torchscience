import torch
from torch import Tensor

import torchscience

__all__ = ["carlson_elliptic_integral_r_c"]


def carlson_elliptic_integral_r_c(x: Tensor, y: Tensor) -> Tensor:
    r"""
    Carlson's degenerate elliptic integral :math:`R_C(x, y)`.

    .. math::
        R_C(x, y) = \frac{1}{2} \int_0^{\infty} \frac{dt}{(t + x)^{1/2}(t + y)}

    This is a special case of Carlson's elliptic integral :math:`R_F`:

    .. math::
        R_C(x, y) = R_F(x, y, y)

    Args:
        x (Tensor): First argument (must be non-negative).
        y (Tensor): Second argument (must be positive).

    Returns:
        Tensor: The value of Carlson's degenerate elliptic integral.
    """
    return torchscience.ops.torchscience._carlson_elliptic_integral_r_c(x, y)
