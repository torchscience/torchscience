import torch
from torch import Tensor


def binomial_coefficient(n: Tensor, k: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Computes the binomial coefficient "n choose k".

    .. math::

        \\text{binomial\\_coefficient}(n, k) = \\binom{n}{k} = \\frac{n!}{k!(n-k)!}

    The number of ways to choose k items from n items without repetition and
    without order.

    Parameters
    ----------
    n : Tensor
        Total number of items (non-negative integer).

    k : Tensor
        Number of items to choose (non-negative integer, k <= n).

    out : Tensor, optional
        Output tensor.

    Returns
    -------
    Tensor
        The binomial coefficient for each element pair.

    Examples
    --------
    >>> binomial_coefficient(torch.tensor([5.0, 6.0, 10.0]), torch.tensor([2.0, 3.0, 5.0]))
    tensor([ 10.,  20., 252.])

    >>> # C(5,2) = 10, C(6,3) = 20, C(10,5) = 252
    """
    output: Tensor = torch.ops.torchscience._binomial_coefficient(n, k)

    if out is not None:
        out.copy_(output)

        return out

    return output
