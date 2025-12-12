import torch
from torch import Tensor


def double_factorial(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Computes the double factorial of each element in the input tensor.

    .. math::

        \\text{double\\_factorial}(n) = n!! = \\begin{cases}
            n \\times (n-2) \\times \\cdots \\times 3 \\times 1 & \\text{if } n \\text{ is odd} \\\\
            n \\times (n-2) \\times \\cdots \\times 4 \\times 2 & \\text{if } n \\text{ is even}
        \\end{cases}

    By convention, :math:`0!! = 1` and :math:`(-1)!! = 1`.

    Parameters
    ----------
    input : Tensor
        Input tensor containing non-negative integers.

    out : Tensor, optional
        Output tensor.

    Returns
    -------
    Tensor
        The double factorial of each element in the input tensor.

    Examples
    --------
    >>> double_factorial(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    tensor([  1.,   1.,   2.,   3.,   8.,  15.,  48.])
    """
    output: Tensor = torch.ops.torchscience._double_factorial(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
