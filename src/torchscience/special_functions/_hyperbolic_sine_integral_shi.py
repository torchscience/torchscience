from torch import Tensor

import torch


def hyperbolic_sine_integral_shi(input: Tensor, *, out: Tensor | None = None) -> Tensor:
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
    output = torch.ops.torchscience._hyperbolic_sine_integral_shi(input)

    if out is not None:
        out.copy_(output)
        return out

    return output
