import torch
from torch import Tensor


def bessel_y_derivative(nu: Tensor, x: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Derivative of the Bessel function of the second kind.

    .. math::

        Y'_\\nu(x) = \\frac{Y_{\\nu-1}(x) - Y_{\\nu+1}(x)}{2}

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
        The derivative of Y_nu evaluated at each element pair.

    Examples
    --------
    >>> bessel_y_derivative(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 2.0]))
    tensor([ 0.7812, -0.1173])
    """
    output: Tensor = torch.ops.torchscience._bessel_y_derivative(nu, x)

    if out is not None:
        out.copy_(output)

        return out

    return output
