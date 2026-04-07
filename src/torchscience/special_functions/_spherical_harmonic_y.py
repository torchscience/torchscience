import torch
from torch import Tensor


def spherical_harmonic_y(
    l: Tensor, m: Tensor, theta: Tensor, phi: Tensor
) -> Tensor:
    r"""
    Spherical harmonic.

    Computes the spherical harmonic :math:`Y_l^m(\theta, \phi)`.

    Mathematical Definition
    -----------------------
    The complex spherical harmonic is defined as:

    .. math::

        Y_l^m(\theta, \phi) = N_l^m P_l^m(\cos\theta) e^{im\phi}

    where the normalization factor is:

    .. math::

        N_l^m = \sqrt{\frac{(2l+1)}{4\pi} \frac{(l-m)!}{(l+m)!}}

    and :math:`P_l^m` is the associated Legendre polynomial with the
    Condon-Shortley phase convention.

    For negative :math:`m`:

    .. math::

        Y_l^{-|m|} = (-1)^m \overline{Y_l^{|m|}}

    Special Values
    --------------
    - :math:`Y_0^0 = \frac{1}{\sqrt{4\pi}}` for all :math:`(\theta, \phi)`
    - :math:`Y_l^0 = \sqrt{\frac{2l+1}{4\pi}} P_l(\cos\theta)`

    Applications
    ------------
    - **Quantum mechanics**: Angular momentum eigenfunctions, atomic
      orbitals, and angular part of the hydrogen atom wavefunctions.

    - **Computer graphics**: Environment lighting, precomputed radiance
      transfer (PRT), and spherical function representation.

    - **Geophysics**: Gravitational and magnetic field modeling via
      spherical harmonic expansion.

    - **Electrostatics**: Multipole expansion of charge distributions.

    Parameters
    ----------
    l : Tensor
        Degree of the spherical harmonic. Must be a non-negative integer.
        Broadcasting with m, theta, and phi is supported.
    m : Tensor
        Order of the spherical harmonic. Must satisfy :math:`|m| \leq l`.
        Broadcasting with l, theta, and phi is supported.
    theta : Tensor
        Polar angle (colatitude) in radians, range :math:`[0, \pi]`.
        Broadcasting with l, m, and phi is supported.
    phi : Tensor
        Azimuthal angle in radians, range :math:`[0, 2\pi]`.
        Broadcasting with l, m, and theta is supported.

    Returns
    -------
    Tensor
        The spherical harmonic :math:`Y_l^m(\theta, \phi)` evaluated at the
        input values.

    Examples
    --------
    Y_0^0 is constant:

    >>> import math
    >>> l = torch.tensor([0.0])
    >>> m = torch.tensor([0.0])
    >>> theta = torch.tensor([0.0])
    >>> phi = torch.tensor([0.0])
    >>> result = spherical_harmonic_y(l, m, theta, phi)

    .. warning:: Gradients use finite differences

       The gradients with respect to l and m are zero (discrete parameters).
       Second-order gradients use finite differences and may have reduced
       accuracy compared to analytical gradients.

    See Also
    --------
    associated_legendre_polynomial_p : Associated Legendre polynomials
    """
    # Promote all inputs to complex dtype (spherical harmonics are complex-valued)
    dtype = torch.promote_types(
        torch.result_type(l, m), torch.result_type(theta, phi)
    )

    if not dtype.is_complex:
        dtype = torch.complex128 if dtype == torch.float64 else torch.complex64

    return torch.ops.torchscience.spherical_harmonic_y(
        l.to(dtype), m.to(dtype), theta.to(dtype), phi.to(dtype)
    )
