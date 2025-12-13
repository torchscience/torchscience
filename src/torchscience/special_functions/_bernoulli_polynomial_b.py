from torch import Tensor

import torchscience.ops.torchscience

__all__ = ["bernoulli_polynomial_b"]


def bernoulli_polynomial_b(n: Tensor, x: Tensor) -> Tensor:
    r"""
    Bernoulli polynomial :math:`B_n(x)`.

    The Bernoulli polynomials are defined by:

    .. math::
        B_n(x) = \sum_{k=0}^{n} \binom{n}{k} B_k x^{n-k}

    where :math:`B_k` are the Bernoulli numbers.

    Special cases:

    - :math:`B_0(x) = 1`
    - :math:`B_1(x) = x - \frac{1}{2}`
    - :math:`B_2(x) = x^2 - x + \frac{1}{6}`

    The derivative satisfies :math:`\frac{d}{dx} B_n(x) = n B_{n-1}(x)`.

    Parameters
    ----------
    n : Tensor
        Non-negative integer degree of the polynomial.
    x : Tensor
        Argument at which to evaluate the polynomial.

    Returns
    -------
    Tensor
        The Bernoulli polynomial :math:`B_n(x)`.
    """
    return torchscience.ops.torchscience._bernoulli_polynomial_b(n, x)
