import torch
from torch import Tensor


def sinc_pi(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Normalized sinc function.

    .. math::

        \\text{sinc}_\\pi(x) = \\frac{\\sin(\\pi x)}{\\pi x}

    with the limiting value ``sinc_pi(0) = 1``.

    This is the normalized sinc function commonly used in signal processing
    and Fourier analysis. It satisfies ``sinc_pi(n) = 0`` for all nonzero
    integers ``n``.

    Properties:

    - ``sinc_pi(0) = 1``
    - ``sinc_pi(n) = 0`` for nonzero integer ``n``
    - ``sinc_pi(-x) = sinc_pi(x)`` (even function)
    - Bounded: ``|sinc_pi(x)| <= 1``

    Parameters
    ----------
    input : Tensor, shape=(...)
        Input tensor.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The normalized sinc function evaluated at each element of `input`.

    Examples
    --------
    >>> sinc_pi(torch.tensor([0.0, 0.5, 1.0, 2.0]))
    tensor([ 1.0000,  0.6366,  0.0000,  0.0000])

    >>> sinc_pi(torch.tensor([-0.5, 0.5]))
    tensor([0.6366, 0.6366])

    Notes
    -----
    The normalized sinc function is related to the unnormalized sinc by:

    .. math::

        \\text{sinc}_\\pi(x) = \\text{sinc}(\\pi x) = \\frac{\\sin(\\pi x)}{\\pi x}

    This is the Fourier transform of the rectangular function and is
    fundamental in sampling theory (Shannon-Nyquist theorem).
    """
    output: Tensor = torch.ops.torchscience._sinc_pi(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
