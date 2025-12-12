import torch
from torch import Tensor


def bessel_j_derivative(nu: Tensor, x: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Derivative of the Bessel function of the first kind.

    .. math::

        J'_\\nu(x) = \\frac{J_{\\nu-1}(x) - J_{\\nu+1}(x)}{2}

    Parameters
    ----------
    nu : Tensor, shape=(...)
        Order of the Bessel function.

    x : Tensor, shape=(...)
        Input tensor.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The derivative of J_nu evaluated at each element pair.

    Examples
    --------
    >>> bessel_j_derivative(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 2.0]))
    tensor([-0.4401,  0.2346])
    """
    output: Tensor = torch.ops.torchscience._bessel_j_derivative(nu, x)

    if out is not None:
        out.copy_(output)

        return out

    return output
