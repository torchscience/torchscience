import torch
from torch import Tensor


def fibonacci_number_f(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Computes the nth Fibonacci number for each element in the input tensor.

    .. math::

        F_n

    where :math:`F_n` is the nth Fibonacci number, defined by the recurrence
    :math:`F_n = F_{n-1} + F_{n-2}` with :math:`F_0 = 0` and :math:`F_1 = 1`.

    Parameters
    ----------
    input : Tensor
        Input tensor containing non-negative integers n for which F_n is computed.

    out : Tensor, optional
        Output tensor.

    Returns
    -------
    Tensor
        The nth Fibonacci number for each element in the input tensor.

    Examples
    --------
    >>> fibonacci_number_f(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    tensor([ 0.,  1.,  1.,  2.,  3.,  5.,  8.])
    """
    output: Tensor = torch.ops.torchscience._fibonacci_number_f(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
