from torch import Tensor

from torchscience._C import _bulirsch_elliptic_integral_el1


def bulirsch_elliptic_integral_el1(x: Tensor, kc: Tensor) -> Tensor:
    r"""
    Bulirsch's incomplete elliptic integral of the first kind.

    .. math::
        \text{el1}(x, k_c) = x \cdot R_F(1, 1 + k_c^2 x^2, 1 + x^2)

    where :math:`R_F` is Carlson's elliptic integral and :math:`k_c` is
    the complementary modulus.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    kc : Tensor
        Complementary modulus.

    Returns
    -------
    Tensor
        Bulirsch's incomplete elliptic integral of the first kind.
    """
    return _bulirsch_elliptic_integral_el1(x, kc)
