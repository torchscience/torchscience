import torch
from torch import Tensor


def bernoulli_number_b(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Computes the Bernoulli number B_2n for each element in the input tensor.

    .. math::

        B_{2n}

    The Bernoulli numbers are a sequence of rational numbers with deep
    connections to number theory. This function returns B_2n (the even-indexed
    Bernoulli numbers), as odd Bernoulli numbers (except B_1) are zero.

    Parameters
    ----------
    input : Tensor
        Input tensor containing non-negative integers n for which B_2n is computed.

    out : Tensor, optional
        Output tensor.

    Returns
    -------
    Tensor
        The Bernoulli number B_2n for each element in the input tensor.

    Examples
    --------
    >>> bernoulli_number_b(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]))
    tensor([ 1.0000, -0.1667,  0.0333, -0.0238,  0.0333])
    """
    output: Tensor = torch.ops.torchscience._bernoulli_number_b(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
