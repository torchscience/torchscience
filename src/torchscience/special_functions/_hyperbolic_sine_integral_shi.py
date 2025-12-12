from torch import Tensor

from torchscience._C import _hyperbolic_sine_integral_shi


def hyperbolic_sine_integral_shi(input: Tensor) -> Tensor:
    r"""
    Hyperbolic sine integral.

    .. math::
        \operatorname{Shi}(x) = \int_0^x \frac{\sinh(t)}{t} \, dt

    Parameters
    ----------
    input : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Hyperbolic sine integral of input.
    """
    return _hyperbolic_sine_integral_shi(input)
