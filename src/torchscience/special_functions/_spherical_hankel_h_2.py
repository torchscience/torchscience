from torch import Tensor

from torchscience._C import _spherical_hankel_h_2


def spherical_hankel_h_2(n: Tensor, x: Tensor) -> Tensor:
    r"""
    Spherical Hankel function of the second kind.

    .. math::
        h^{(2)}_n(x) = j_n(x) - i y_n(x)

    where :math:`j_n` is the spherical Bessel function of the first kind and
    :math:`y_n` is the spherical Bessel function of the second kind.

    For real-valued inputs, returns the magnitude :math:`|h^{(2)}_n(x)|`.
    For complex-valued inputs, returns the full complex spherical Hankel function.

    Parameters
    ----------
    n : Tensor
        Order of the spherical Hankel function (non-negative integer).
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Spherical Hankel function of the second kind.
    """
    return _spherical_hankel_h_2(n, x)
