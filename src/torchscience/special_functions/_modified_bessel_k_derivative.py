import torch
from torch import Tensor


def modified_bessel_k_derivative(nu: Tensor, x: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Derivative of the modified Bessel function of the second kind.

    .. math::

        K'_\\nu(x) = -\\frac{K_{\\nu-1}(x) + K_{\\nu+1}(x)}{2}

    Parameters
    ----------
    nu : Tensor, shape=(...)
        Order of the modified Bessel function.

    x : Tensor, shape=(...)
        Input tensor.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The derivative of K_nu evaluated at each element pair.

    Examples
    --------
    >>> modified_bessel_k_derivative(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 2.0]))
    tensor([-1.0229, -0.1649])
    """
    output: Tensor = torch.ops.torchscience._modified_bessel_k_derivative(nu, x)

    if out is not None:
        out.copy_(output)

        return out

    return output
