from torch import Tensor

import torch


def hyperbolic_cosine_integral_chi(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    r"""
    Hyperbolic cosine integral.

    .. math::
        \operatorname{Chi}(x) = \gamma + \ln|x| + \int_0^x \frac{\cosh(t) - 1}{t} \, dt

    where :math:`\gamma` is the Euler-Mascheroni constant.

    Parameters
    ----------
    input : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Hyperbolic cosine integral of input.
    """
    output = torch.ops.torchscience._hyperbolic_cosine_integral_chi(input)

    if out is not None:
        out.copy_(output)
        return out

    return output
