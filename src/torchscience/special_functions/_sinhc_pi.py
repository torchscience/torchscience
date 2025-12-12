import torch
from torch import Tensor


def sinhc_pi(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Hyperbolic sinc function (normalized).

    .. math::

        \\text{sinhc}_\\pi(x) = \\frac{\\sinh(\\pi x)}{\\pi x}

    with the limiting value ``sinhc_pi(0) = 1``.

    This is the hyperbolic analog of the normalized sinc function.

    Properties:

    - ``sinhc_pi(0) = 1``
    - ``sinhc_pi(-x) = sinhc_pi(x)`` (even function)
    - ``sinhc_pi(x) >= 1`` for all real ``x``
    - Grows exponentially: ``sinhc_pi(x) ~ exp(pi*|x|) / (2*pi*|x|)`` for large ``|x|``

    Parameters
    ----------
    input : Tensor, shape=(...)
        Input tensor.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The hyperbolic sinc function evaluated at each element of `input`.

    Examples
    --------
    >>> sinhc_pi(torch.tensor([0.0, 0.5, 1.0]))
    tensor([1.0000, 1.5924, 3.6769])

    >>> sinhc_pi(torch.tensor([-1.0, 1.0]))
    tensor([3.6769, 3.6769])

    Notes
    -----
    The hyperbolic sinc function is related to the regular sinc function by
    replacing the trigonometric sine with the hyperbolic sine:

    .. math::

        \\text{sinhc}_\\pi(x) = \\frac{\\sinh(\\pi x)}{\\pi x}

    Unlike the regular sinc function which oscillates and decays,
    sinhc_pi grows exponentially for large |x|.
    """
    output: Tensor = torch.ops.torchscience._sinhc_pi(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
