from torch import Tensor

from torchscience._C import _hyperbolic_cosine_integral_chi


def hyperbolic_cosine_integral_chi(input: Tensor) -> Tensor:
    r"""
    Hyperbolic cosine integral.

    .. math::
        \operatorname{Chi}(x) = \gamma + \ln|x| + \int_0^x \frac{\cosh(t) - 1}{t} \, dt

    where :math:`\gamma` is the Euler-Mascheroni constant.

    Parameters
    ----------
    input : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Hyperbolic cosine integral of input.
    """
    return _hyperbolic_cosine_integral_chi(input)
