import torch
from torch import Tensor


def logarithmic_integral_li(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Logarithmic integral function.

    .. math::

        \\text{li}(x) = \\int_0^x \\frac{dt}{\\ln t}

    The logarithmic integral is the principal value of the integral.
    It is related to the exponential integral by:

    .. math::

        \\text{li}(x) = \\text{Ei}(\\ln x)

    Parameters
    ----------
    input : Tensor, shape=(...)
        Input tensor. Values must be positive.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The logarithmic integral evaluated at each element of `input`.

    Examples
    --------
    >>> logarithmic_integral_li(torch.tensor([2.0, 3.0, 10.0]))
    tensor([1.0451, 2.1635, 6.1656])

    Notes
    -----
    The logarithmic integral is important in number theory, particularly
    in the prime number theorem where li(x) approximates the number of
    primes less than x.
    """
    output: Tensor = torch.ops.torchscience._logarithmic_integral_li(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
