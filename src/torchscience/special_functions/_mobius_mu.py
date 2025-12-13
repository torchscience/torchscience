import torch
from torch import Tensor


def mobius_mu(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    r"""
    Mobius function :math:`\mu(n)`.

    The Mobius function is a multiplicative function defined as:

    - :math:`\mu(1) = 1`
    - :math:`\mu(n) = (-1)^k` if n is a product of k distinct primes (square-free)
    - :math:`\mu(n) = 0` if n has a squared prime factor

    Properties:

    - Multiplicative: :math:`\mu(mn) = \mu(m)\mu(n)` when :math:`\gcd(m,n) = 1`
    - :math:`\sum_{d | n} \mu(d) = [n = 1]` (1 if n=1, else 0)
    - Used in the Mobius inversion formula

    Parameters
    ----------
    input : Tensor
        Input tensor containing positive integers n.

    out : Tensor, optional
        Output tensor.

    Returns
    -------
    Tensor
        The Mobius function :math:`\mu(n)` for each element.

    Examples
    --------
    >>> mobius_mu(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    tensor([ 1., -1., -1.,  0., -1.,  1.])
    """
    output: Tensor = torch.ops.torchscience._mobius_mu(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
