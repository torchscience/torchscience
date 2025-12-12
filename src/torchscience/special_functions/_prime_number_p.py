import torch
from torch import Tensor


def prime_number_p(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Computes the nth prime number for each element in the input tensor.

    .. math::

        p_n

    where :math:`p_n` is the nth prime number (0-indexed, so :math:`p_0 = 2`).

    Parameters
    ----------
    input : Tensor
        Input tensor containing non-negative integers n for which the nth prime
        is computed.

    out : Tensor, optional
        Output tensor.

    Returns
    -------
    Tensor
        The nth prime number for each element in the input tensor.

    Examples
    --------
    >>> prime_number_p(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]))
    tensor([ 2.,  3.,  5.,  7., 11.])
    """
    output: Tensor = torch.ops.torchscience._prime_number_p(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
