import torch
from torch import Tensor


def sin_pi(x: Tensor) -> Tensor:
    r"""
    Sine of pi times x.

    Computes :math:`\sin(\pi x)` with higher accuracy than ``torch.sin(math.pi * x)``
    for values near integers where :math:`\sin(\pi n) = 0`.

    Parameters
    ----------
    x : Tensor
        Input tensor. Supports complex tensors.

    Returns
    -------
    Tensor
        The value of sin(pi * x).

    See Also
    --------
    cos_pi : Cosine of pi times x
    """
    return torch.ops.torchscience.sin_pi(x)
