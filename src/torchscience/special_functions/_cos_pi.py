import torch
from torch import Tensor


def cos_pi(x: Tensor) -> Tensor:
    r"""
    Cosine of pi times x.

    Computes :math:`\cos(\pi x)` with higher accuracy than ``torch.cos(math.pi * x)``
    for values near integers where :math:`\cos(\pi n) = (-1)^n`.

    Parameters
    ----------
    x : Tensor
        Input tensor. Supports complex tensors.

    Returns
    -------
    Tensor
        The value of cos(pi * x).

    See Also
    --------
    sin_pi : Sine of pi times x
    """
    return torch.ops.torchscience.cos_pi(x)
