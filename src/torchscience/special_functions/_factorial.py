import torch
from torch import Tensor


def factorial(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Computes the factorial of each element in the input tensor.

    .. math::

        \\text{factorial}(n) = n! = n \\times (n-1) \\times \\cdots \\times 2 \\times 1

    For non-negative integers, :math:`n! = \\prod_{k=1}^{n} k`.

    By convention, :math:`0! = 1`.

    Parameters
    ----------
    input : Tensor
        Input tensor containing non-negative integers.

    out : Tensor, optional
        Output tensor.

    Returns
    -------
    Tensor
        The factorial of each element in the input tensor.

    Examples
    --------
    >>> factorial(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
    tensor([  1.,   1.,   2.,   6.,  24., 120.])
    """
    output: Tensor = torch.ops.torchscience._factorial(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
