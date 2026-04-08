import torch
from torch import Tensor


def tan_pi(x: Tensor) -> Tensor:
    r"""
    Tangent of pi times x.

    Computes :math:`\tan(\pi x)` with higher accuracy than ``torch.tan(math.pi * x)``
    for values near half-integers where :math:`\tan(\pi x)` has poles.

    Parameters
    ----------
    x : Tensor
        Input tensor. Poles at half-integers. Supports complex tensors.

    Returns
    -------
    Tensor
        The value of tan(pi * x).

    See Also
    --------
    sin_pi : Sine of pi times x
    cos_pi : Cosine of pi times x
    """
    return torch.ops.torchscience.tan_pi(x)
