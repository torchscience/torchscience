import torch
from torch import Tensor


def cosh_pi(x: Tensor) -> Tensor:
    r"""
    Hyperbolic cosine of pi times x.

    Computes :math:`\cosh(\pi x)`.

    Parameters
    ----------
    x : Tensor
        Input tensor. Supports complex tensors.

    Returns
    -------
    Tensor
        The value of cosh(pi * x).

    See Also
    --------
    sinh_pi : Hyperbolic sine of pi times x
    """
    return torch.ops.torchscience.cosh_pi(x)
