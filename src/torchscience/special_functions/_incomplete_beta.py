import torch
from torch import Tensor


def incomplete_beta(z: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """
    Regularized incomplete beta function.

    Computes the regularized incomplete beta function I_z(a, b).

    Mathematical Definition
    -----------------------
    The regularized incomplete beta function is defined as:

        I_z(a, b) = B_z(a, b) / B(a, b)

    where:
        - B_z(a, b) = integral from 0 to z of t^(a-1) * (1-t)^(b-1) dt
          is the incomplete beta function
        - B(a, b) = Gamma(a) * Gamma(b) / Gamma(a+b) is the beta function

    The function satisfies the symmetry relation:
        I_z(a, b) + I_{1-z}(b, a) = 1

    Domain
    ------
    - z: Real z in [0, 1], or complex z (extended domain via analytic continuation)
    - a: must be positive (a > 0, or Re(a) > 0 for complex)
    - b: must be positive (b > 0, or Re(b) > 0 for complex)

    For real z outside [0, 1] or non-positive a, b, the result is NaN.

    For complex z with |z| >= 1, uses analytic continuation via the
    hypergeometric 2F1 relation:
        I_z(a, b) = z^a / (a * B(a,b)) * 2F1(a, 1-b; a+1; z)

    Two regions are handled:
    - Region B: |z| >= 1, |1-z| < 1 -- uses symmetry I_z(a,b) = 1 - I_{1-z}(b,a)
    - Region C: |z| > 1, |1-z| >= 1 -- uses hypergeometric linear transformation

    Algorithm
    ---------
    Uses a continued fraction expansion for numerical evaluation:
    - When z < (a + 1) / (a + b + 2), uses direct continued fraction
    - Otherwise, uses symmetry: I_z(a, b) = 1 - I_{1-z}(b, a)

    The continued fraction is evaluated using a modified Lentz's algorithm
    for numerical stability.

    Special Values
    --------------
    - I_0(a, b) = 0
    - I_1(a, b) = 1
    - I_z(1, 1) = z (uniform distribution CDF)
    - I_z(1, b) = 1 - (1-z)^b
    - I_z(a, 1) = z^a

    Applications
    ------------
    The regularized incomplete beta function is widely used in statistics:
    - Cumulative distribution function (CDF) of the beta distribution
    - CDF of the binomial distribution
    - CDF of the F-distribution
    - CDF of Student's t-distribution
    - Confidence intervals for binomial proportions

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype
    - Supports float16, bfloat16, float32, float64
    - Complex dtypes (complex64, complex128) are supported for all z via
      analytic continuation

    Autograd Support
    ----------------
    All first-order derivatives are computed analytically:
    - dI/dz = z^(a-1) * (1-z)^(b-1) / B(a, b)
    - dI/da = J_a / B(a,b) - I_z(a,b) * [psi(a) - psi(a+b)]
    - dI/db = J_b / B(a,b) - I_z(a,b) * [psi(b) - psi(a+b)]

    where psi is the digamma function and J_a, J_b are log-weighted
    integrals computed using 32-point Gauss-Legendre quadrature.

    All second-order derivatives are also fully analytical using:
    - Trigamma functions psi'(x) for parameter second derivatives
    - Doubly log-weighted integrals K_aa, K_ab, K_bb via quadrature

    The analytical formulas for parameter second derivatives are:
    - d²I/da² = K_aa/B - 2(J_a/B)(psi(a)-psi(a+b)) + I_z(psi(a)-psi(a+b))²
              - I_z(psi'(a) - psi'(a+b))
    - d²I/db² = K_bb/B - 2(J_b/B)(psi(b)-psi(a+b)) + I_z(psi(b)-psi(a+b))²
              - I_z(psi'(b) - psi'(a+b))
    - d²I/dadb = K_ab/B - (J_a/B)(psi(b)-psi(a+b)) - (J_b/B)(psi(a)-psi(a+b))
               + I_z(psi(a)-psi(a+b))(psi(b)-psi(a+b)) + I_z*psi'(a+b)

    For complex inputs, Wirtinger derivative conventions are used:
    - First backward: grad_x = grad_output * conj(∂f/∂x)
    - Double backward: ∂(grad_x)/∂x̄ = grad_output * conj(∂²f/∂x²)

    Parameters
    ----------
    z : Tensor
        Input tensor. For real dtypes, values should be in [0, 1].
        For complex dtypes, values can be anywhere in the complex plane
        (analytic continuation is used for |z| >= 1).
        Broadcasting with a and b is supported.
    a : Tensor
        First shape parameter. Must be positive (a > 0 for real, Re(a) > 0 for complex).
        Broadcasting with z and b is supported.
    b : Tensor
        Second shape parameter. Must be positive (b > 0 for real, Re(b) > 0 for complex).
        Broadcasting with z and a is supported.

    Returns
    -------
    Tensor
        The regularized incomplete beta function I_z(a, b) evaluated at the
        input values. Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> z = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    >>> a = torch.tensor([2.0])
    >>> b = torch.tensor([5.0])
    >>> incomplete_beta(z, a, b)
    tensor([0.0000, 0.6328, 0.8906, 0.9844, 1.0000])

    Special case I_z(1, 1) = z (uniform distribution):

    >>> z = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    >>> a = torch.tensor([1.0])
    >>> b = torch.tensor([1.0])
    >>> incomplete_beta(z, a, b)
    tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])

    Beta distribution CDF:

    >>> z = torch.tensor([0.5])
    >>> a = torch.tensor([2.0])
    >>> b = torch.tensor([2.0])
    >>> incomplete_beta(z, a, b)  # Symmetric beta, CDF at 0.5 should be 0.5
    tensor([0.5000])

    Autograd example:

    >>> z = torch.tensor([0.3], requires_grad=True)
    >>> a = torch.tensor([2.0])
    >>> b = torch.tensor([3.0])
    >>> y = incomplete_beta(z, a, b)
    >>> y.backward()
    >>> z.grad  # Gradient w.r.t. z
    tensor([1.7640])

    Warnings
    --------
    - For z very close to 0 or 1, numerical precision may be reduced
    - For very large or very small values of a and b, the continued
      fraction may converge slowly or lose precision
    - For extreme parameter values (a or b near 0 or very large), the
      log-weighted integrals used for gradients may lose accuracy
    - For complex z with |z| > 1 in Region C (where |1-z| >= 1), gradients
      use finite differences and may have reduced accuracy compared to
      the analytical gradients available in other regions
    - When a - b is close to an integer, the hypergeometric continuation
      uses a regularization technique which may have slightly reduced accuracy
    """
    return torch.ops.torchscience.incomplete_beta(z, a, b)
