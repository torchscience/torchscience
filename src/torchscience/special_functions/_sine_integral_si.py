import torch
from torch import Tensor


def sine_integral_si(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Sine integral function Si(x).

    .. math::

        \\text{Si}(x) = \\int_0^x \\frac{\\sin t}{t} dt

    The sine integral is the antiderivative of sinc(x) = sin(x)/x.

    Parameters
    ----------
    input : Tensor, shape=(...)
        Input tensor.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The sine integral evaluated at each element of `input`.

    Examples
    --------
    >>> sine_integral_si(torch.tensor([0.0, 1.0, 2.0]))
    tensor([0.0000, 0.9461, 1.6054])

    Notes
    -----
    The sine integral Si(x) is related to the shifted sine integral si(x) by
    Si(x) = si(x) + pi/2. The asymptotic behavior is Si(x) -> pi/2 as x -> infinity.
    """
    output: Tensor = torch.ops.torchscience._sine_integral_si(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
