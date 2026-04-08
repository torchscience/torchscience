import torch
from torch import Tensor


def tanh_pi(x: Tensor) -> Tensor:
    r"""
    Hyperbolic tangent of pi times x.

    Computes :math:`\tanh(\pi x)`.

    Parameters
    ----------
    x : Tensor
        Input tensor. Supports complex tensors.

    Returns
    -------
    Tensor
        The value of tanh(pi * x).

    See Also
    --------
    sinh_pi : Hyperbolic sine of pi times x
    cosh_pi : Hyperbolic cosine of pi times x
    """
    return torch.ops.torchscience.tanh_pi(x)
