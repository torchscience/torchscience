from torch import Tensor

from torchscience._C import _hankel_h_1


def hankel_h_1(nu: Tensor, x: Tensor) -> Tensor:
    r"""
    Hankel function of the first kind.

    .. math::
        H^{(1)}_\nu(x) = J_\nu(x) + i Y_\nu(x)

    where :math:`J_\nu` is the Bessel function of the first kind and
    :math:`Y_\nu` is the Bessel function of the second kind.

    For real-valued inputs, returns the magnitude :math:`|H^{(1)}_\nu(x)|`.
    For complex-valued inputs, returns the full complex Hankel function.

    Parameters
    ----------
    nu : Tensor
        Order of the Hankel function.
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Hankel function of the first kind.
    """
    return _hankel_h_1(nu, x)
