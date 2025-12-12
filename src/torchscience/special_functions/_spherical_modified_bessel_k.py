import torch
from torch import Tensor


def spherical_modified_bessel_k(n: Tensor, x: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Spherical modified Bessel function of the second kind.

    .. math::

        k_n(x) = \\sqrt{\\frac{\\pi}{2x}} K_{n+1/2}(x)

    Parameters
    ----------
    n : Tensor, shape=(...)
        Order of the spherical modified Bessel function.

    x : Tensor, shape=(...)
        Argument of the spherical modified Bessel function.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The spherical modified Bessel function of the second kind evaluated at each element pair.

    Notes
    -----
    Gradients with respect to the order `n` are not supported and will be zero.
    Gradients with respect to `x` use the recurrence relation:

    .. math::

        \\frac{d}{dx} k_n(x) = -k_{n-1}(x) - \\frac{n+1}{x} k_n(x)

    Examples
    --------
    >>> spherical_modified_bessel_k(torch.tensor([0.0]), torch.tensor([1.0]))
    tensor([0.4658])

    >>> spherical_modified_bessel_k(torch.tensor([0.0, 1.0, 2.0]), torch.tensor([1.0, 1.0, 1.0]))
    tensor([0.4658, 0.7854, 1.8750])
    """
    output: Tensor = torch.ops.torchscience._spherical_modified_bessel_k(n, x)

    if out is not None:
        out.copy_(output)

        return out

    return output
