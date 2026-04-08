import torch
from torch import Tensor


def associated_legendre_polynomial_p(
    n: Tensor, m: Tensor, x: Tensor
) -> Tensor:
    r"""
    Associated Legendre polynomial of the first kind.

    Computes the associated Legendre polynomial :math:`P_n^m(x)` using the
    Condon-Shortley phase convention.

    Mathematical Definition
    -----------------------
    The associated Legendre polynomial is defined as:

    .. math::

        P_n^m(x) = (-1)^m (1-x^2)^{m/2} \frac{d^m}{dx^m} P_n(x)

    where :math:`P_n(x)` is the Legendre polynomial of degree n.

    For negative m, the symmetry relation is used:

    .. math::

        P_n^{-|m|}(x) = (-1)^{|m|} \frac{(n-|m|)!}{(n+|m|)!} P_n^{|m|}(x)

    Special Values
    --------------
    - :math:`P_0^0(x) = 1`
    - :math:`P_1^0(x) = x`
    - :math:`P_1^1(x) = -\sqrt{1-x^2}`
    - :math:`P_2^0(x) = (3x^2-1)/2`
    - :math:`P_n^m(x) = 0` when :math:`|m| > n`

    Domain
    ------
    - n: integer-valued tensor (degree, non-negative)
    - m: integer-valued tensor (order, :math:`|m| \leq n`)
    - x: real tensor in [-1, 1]

    Algorithm
    ---------
    Uses three-term recurrence for numerical stability:

    .. math::

        (k-m+1) P_{k+1}^m(x) = (2k+1) x P_k^m(x) - (k+m) P_{k-1}^m(x)

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype.
    - Supports float16, bfloat16, float32, float64, complex64, complex128.

    Autograd Support
    ----------------
    - Gradients for n and m are zero (discrete parameters).
    - Gradient for x uses the identity:

    .. math::

        \frac{dP_n^m}{dx} = \frac{n x P_n^m(x) - (n+m) P_{n-1}^m(x)}{x^2 - 1}

    - Second-order derivatives are supported.

    Parameters
    ----------
    n : Tensor
        Degree of the polynomial. Integer-valued.
        Broadcasting with m and x is supported.
    m : Tensor
        Order of the polynomial. Integer-valued.
        Broadcasting with n and x is supported.
    x : Tensor
        Input tensor. Values should be in [-1, 1].
        Broadcasting with n and m is supported.

    Returns
    -------
    Tensor
        The associated Legendre polynomial :math:`P_n^m(x)`.

    Examples
    --------
    >>> n = torch.tensor([2.0])
    >>> m = torch.tensor([0.0])
    >>> x = torch.tensor([0.5])
    >>> associated_legendre_polynomial_p(n, m, x)
    tensor([-0.1250])

    >>> n = torch.tensor([1.0])
    >>> m = torch.tensor([1.0])
    >>> x = torch.tensor([0.5])
    >>> associated_legendre_polynomial_p(n, m, x)
    tensor([-0.8660])

    See Also
    --------
    legendre_polynomial_p : Legendre polynomial P_n(x)
    spherical_harmonic_y : Spherical harmonics Y_l^m(theta, phi)
    """
    return torch.ops.torchscience.associated_legendre_polynomial_p(n, m, x)
