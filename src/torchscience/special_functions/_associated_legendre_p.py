from torch import Tensor

import torchscience.ops.torchscience

__all__ = ["associated_legendre_p"]


def associated_legendre_p(n: Tensor, m: Tensor, x: Tensor) -> Tensor:
    r"""
    Associated Legendre polynomial :math:`P_n^m(x)`.

    The associated Legendre polynomials are defined by:

    .. math::
        P_n^m(x) = (-1)^m (1-x^2)^{m/2} \frac{d^m}{dx^m} P_n(x)

    where :math:`P_n(x)` is the Legendre polynomial of degree :math:`n`.

    They satisfy the orthogonality relation:

    .. math::
        \int_{-1}^{1} P_l^m(x) P_k^m(x) dx = \frac{2}{2l+1} \frac{(l+m)!}{(l-m)!} \delta_{lk}

    Parameters
    ----------
    n : Tensor
        Non-negative integer degree (:math:`n \geq 0`).
    m : Tensor
        Integer order (:math:`|m| \leq n`).
    x : Tensor
        Argument, typically in :math:`[-1, 1]`.

    Returns
    -------
    Tensor
        The associated Legendre polynomial :math:`P_n^m(x)`.
    """
    return torchscience.ops.torchscience._associated_legendre_p(n, m, x)
