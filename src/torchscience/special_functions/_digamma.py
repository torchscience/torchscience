import torch
from torch import Tensor


def digamma(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Digamma function (psi function).

    The digamma function is the logarithmic derivative of the gamma function:

    .. math::

        \\psi(x) = \\frac{d}{dx} \\ln\\Gamma(x) = \\frac{\\Gamma'(x)}{\\Gamma(x)}

    Parameters
    ----------
    input : Tensor, shape=(...)
        Input tensor.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The digamma function evaluated at each element of `input`.

    Examples
    --------
    >>> digamma(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    tensor([-0.5772,  0.4228,  0.9228,  1.2561])

    >>> digamma(torch.tensor([0.5]))
    tensor([-1.9635])
    """
    output: Tensor = torch.ops.torchscience._digamma(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
