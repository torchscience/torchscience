import torch
from torch import Tensor


def sinh_pi(x: Tensor) -> Tensor:
    r"""
    Hyperbolic sine of pi times x.

    Computes :math:`\sinh(\pi x)`.

    Parameters
    ----------
    x : Tensor
        Input tensor. Supports complex tensors.

    Returns
    -------
    Tensor
        The value of sinh(pi * x).

    See Also
    --------
    cosh_pi : Hyperbolic cosine of pi times x
    """
    return torch.ops.torchscience.sinh_pi(x)
