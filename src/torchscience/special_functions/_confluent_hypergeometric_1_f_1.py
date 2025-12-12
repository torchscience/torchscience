from torch import Tensor

import torchscience.ops.torchscience


def confluent_hypergeometric_1_f_1(a: Tensor, b: Tensor, z: Tensor) -> Tensor:
    r"""
    Confluent hypergeometric function of the first kind, 1F1(a, b, z).

    Also known as Kummer's function M(a, b, z), it is defined by the series:

    .. math::
        {}_1F_1(a; b; z) = M(a, b, z) = \sum_{n=0}^{\infty} \frac{(a)_n}{(b)_n} \frac{z^n}{n!}

    where :math:`(a)_n` is the Pochhammer symbol (rising factorial).

    Parameters
    ----------
    a : Tensor
        First parameter.
    b : Tensor
        Second parameter (must not be a non-positive integer).
    z : Tensor
        Argument.

    Returns
    -------
    Tensor
        Value of the confluent hypergeometric function 1F1(a, b, z).
    """
    return torchscience.ops.torchscience._confluent_hypergeometric_1_f_1(a, b, z)
