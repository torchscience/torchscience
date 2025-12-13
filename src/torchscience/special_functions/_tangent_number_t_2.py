import torch
from torch import Tensor


def tangent_number_t_2(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Computes the tangent number T_2n for each element in the input tensor.

    .. math::

        T_{2n}

    The tangent numbers appear in the Taylor series expansion of tan(x).
    This function returns T_2n (the even-indexed tangent numbers).

    Parameters
    ----------
    input : Tensor
        Input tensor containing non-negative integers n for which T_2n is computed.

    out : Tensor, optional
        Output tensor.

    Returns
    -------
    Tensor
        The tangent number T_2n for each element in the input tensor.

    Examples
    --------
    >>> tangent_number_t_2(torch.tensor([0.0, 1.0, 2.0, 3.0]))
    tensor([1., 2., 16., 272.])
    """
    output: Tensor = torch.ops.torchscience._tangent_number_t_2(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
