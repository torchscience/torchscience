"""Polynomial root finding."""

import torch
from torch import Tensor

from torchscience.polynomial._exceptions import DegreeError
from torchscience.polynomial._polynomial import Polynomial, polynomial


def polynomial_roots(p: Polynomial) -> Tensor:
    """Find polynomial roots via companion matrix eigenvalues.

    Parameters
    ----------
    p : Polynomial
        Polynomial with coefficients shape (..., N).
        Leading coefficient must be non-zero.

    Returns
    -------
    Tensor
        Complex roots, shape (..., N-1). Always complex dtype.

    Raises
    ------
    DegreeError
        If polynomial is constant (degree 0) or zero polynomial.

    Examples
    --------
    >>> p = polynomial(torch.tensor([2.0, -3.0, 1.0]))  # (x-1)(x-2)
    >>> polynomial_roots(p)
    tensor([1.+0.j, 2.+0.j])

    Notes
    -----
    Uses companion matrix method:
    - Construct companion matrix from normalized coefficients
    - Compute eigenvalues via torch.linalg.eigvals
    - Supports autograd through eigenvalue computation

    For high-degree polynomials (>20), use float64 for accuracy.
    """
    coeffs = p.coeffs
    n = coeffs.shape[-1]  # n = degree + 1

    if n < 2:
        raise DegreeError(
            f"Cannot find roots of constant polynomial (degree 0), got {n} coefficients"
        )

    # Leading coefficient (highest degree)
    leading = coeffs[..., -1]

    # Check for zero leading coefficient
    if torch.any(leading == 0):
        raise DegreeError(
            "Leading coefficient must be non-zero for root finding. "
            "Use polynomial_trim first to remove trailing zeros."
        )

    # Normalize coefficients by leading coefficient
    # normalized[i] = -coeffs[i] / coeffs[-1]
    normalized = -coeffs[..., :-1] / leading.unsqueeze(-1)

    # Construct companion matrix
    # For polynomial p(x) = a_0 + a_1*x + ... + a_{n-1}*x^{n-1} + x^n (monic)
    # Companion matrix is:
    # [[0, 0, ..., 0, -a_0  ],
    #  [1, 0, ..., 0, -a_1  ],
    #  [0, 1, ..., 0, -a_2  ],
    #  [.                   ],
    #  [0, 0, ..., 1, -a_{n-1}]]
    #
    # The eigenvalues of C are the roots of p(x)

    degree = n - 1  # Degree of polynomial = number of roots

    # Handle batch dimensions
    batch_shape = coeffs.shape[:-1]
    batch_size = batch_shape.numel() if len(batch_shape) > 0 else 1

    # Flatten batch dimensions for construction
    if len(batch_shape) > 0:
        normalized_flat = normalized.reshape(batch_size, degree)
    else:
        normalized_flat = normalized.unsqueeze(0)

    # Build companion matrix
    # Start with zeros
    companion = torch.zeros(
        batch_size, degree, degree, dtype=coeffs.dtype, device=coeffs.device
    )

    # Set subdiagonal to 1
    if degree > 1:
        eye_indices = torch.arange(degree - 1, device=coeffs.device)
        companion[:, eye_indices + 1, eye_indices] = 1.0

    # Set last column to normalized coefficients (negated, already done above)
    companion[:, :, -1] = normalized_flat

    # Compute eigenvalues
    # Convert to complex for eigenvalue computation
    companion_complex = companion.to(
        dtype=torch.complex128
        if coeffs.dtype == torch.float64
        else torch.complex64
    )

    roots = torch.linalg.eigvals(companion_complex)

    # Reshape back to batch dimensions
    if len(batch_shape) > 0:
        roots = roots.reshape(*batch_shape, degree)
    else:
        roots = roots.squeeze(0)

    return roots


def polynomial_from_roots(roots: Tensor) -> Polynomial:
    """Construct monic polynomial from its roots.

    Constructs (x - r_0)(x - r_1)...(x - r_{n-1}).

    Parameters
    ----------
    roots : Tensor
        Roots, shape (..., N). Can be complex.

    Returns
    -------
    Polynomial
        Monic polynomial with given roots, shape (..., N+1).

    Examples
    --------
    >>> roots = torch.tensor([1.0, 2.0])  # (x-1)(x-2) = x^2 - 3x + 2
    >>> p = polynomial_from_roots(roots)
    >>> p.coeffs
    tensor([2., -3., 1.])
    """
    # Build polynomial iteratively: start with (x - r_0), multiply by (x - r_i)

    batch_shape = roots.shape[:-1]
    n_roots = roots.shape[-1]

    if n_roots == 0:
        # Empty roots -> constant polynomial 1
        shape = (*batch_shape, 1) if len(batch_shape) > 0 else (1,)
        return polynomial(
            torch.ones(shape, dtype=roots.dtype, device=roots.device)
        )

    # Start with polynomial (x - r_0) = -r_0 + 1*x
    # coeffs = [-r_0, 1]
    if len(batch_shape) > 0:
        ones = torch.ones(batch_shape, dtype=roots.dtype, device=roots.device)
    else:
        ones = torch.ones((), dtype=roots.dtype, device=roots.device)

    coeffs = torch.stack(
        [
            -roots[..., 0],
            ones,
        ],
        dim=-1,
    )

    # Multiply by (x - r_i) for each remaining root
    for i in range(1, n_roots):
        # Current polynomial has degree i, coeffs has shape (..., i+1)
        # Multiply by (x - r_i) = [-r_i, 1]

        root_i = roots[..., i]

        # (c_0 + c_1*x + ... + c_i*x^i) * (x - r_i)
        # = -r_i*c_0 + (-r_i*c_1 + c_0)*x + (-r_i*c_2 + c_1)*x^2 + ... + c_i*x^{i+1}
        # new_coeffs[0] = -r_i * c_0
        # new_coeffs[j] = -r_i * c_j + c_{j-1} for j = 1..i
        # new_coeffs[i+1] = c_i

        # Shift coefficients (multiply by x)
        shifted = torch.nn.functional.pad(coeffs, (1, 0))  # prepend 0

        # Scale original (multiply by -r_i)
        scaled = torch.nn.functional.pad(coeffs, (0, 1)) * (
            -root_i.unsqueeze(-1)
        )

        coeffs = shifted + scaled

    return Polynomial(coeffs=coeffs)


def polynomial_trim(p: Polynomial, tol: float = 0.0) -> Polynomial:
    """Remove trailing near-zero coefficients.

    Parameters
    ----------
    p : Polynomial
        Input polynomial.
    tol : float
        Tolerance for considering coefficient as zero.

    Returns
    -------
    Polynomial
        Trimmed polynomial with at least one coefficient.

    Notes
    -----
    For batched polynomials, this trims based on the maximum absolute
    value across the batch for each coefficient position.
    """
    coeffs = p.coeffs
    n = coeffs.shape[-1]

    if n <= 1:
        return p

    # Find the last non-zero coefficient
    # For batched case, a coefficient position is non-zero if any batch element is non-zero
    if coeffs.dim() > 1:
        abs_coeffs = coeffs.abs()
        # Max over batch dimensions
        max_abs = abs_coeffs
        for _ in range(coeffs.dim() - 1):
            max_abs = max_abs.max(dim=0).values
    else:
        max_abs = coeffs.abs()

    # Find last position > tol
    mask = max_abs > tol
    if not mask.any():
        # All zeros, return single zero coefficient
        return Polynomial(
            coeffs=torch.zeros(
                *coeffs.shape[:-1], 1, dtype=coeffs.dtype, device=coeffs.device
            )
        )

    # Find last True position
    indices = torch.arange(n, device=coeffs.device)
    last_nonzero = indices[mask].max().item()

    # Keep coefficients up to and including last_nonzero
    return Polynomial(coeffs=coeffs[..., : last_nonzero + 1])


def polynomial_equal(
    p: Polynomial, q: Polynomial, tol: float = 1e-8
) -> Tensor:
    """Check polynomial equality within tolerance.

    Parameters
    ----------
    p, q : Polynomial
        Polynomials to compare.
    tol : float
        Absolute tolerance for coefficient comparison.

    Returns
    -------
    Tensor
        Boolean tensor, shape matches broadcast of batch dims.
    """
    p_coeffs = p.coeffs
    q_coeffs = q.coeffs

    # Pad to same length
    n_p = p_coeffs.shape[-1]
    n_q = q_coeffs.shape[-1]

    if n_p < n_q:
        padding = [0] * (2 * (p_coeffs.dim() - 1)) + [0, n_q - n_p]
        p_coeffs = torch.nn.functional.pad(p_coeffs, padding)
    elif n_q < n_p:
        padding = [0] * (2 * (q_coeffs.dim() - 1)) + [0, n_p - n_q]
        q_coeffs = torch.nn.functional.pad(q_coeffs, padding)

    # Check if all coefficients are within tolerance
    diff = (p_coeffs - q_coeffs).abs()

    # All coefficients must be within tolerance
    max_diff = diff.max(dim=-1)
    if isinstance(max_diff, tuple):
        max_diff = max_diff.values
    return max_diff <= tol
