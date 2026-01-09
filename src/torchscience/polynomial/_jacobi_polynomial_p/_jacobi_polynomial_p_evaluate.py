import warnings

import torch
from torch import Tensor

from ._jacobi_polynomial_p import (
    JacobiPolynomialP,
)


def jacobi_polynomial_p_evaluate(
    c: JacobiPolynomialP,
    x: Tensor,
) -> Tensor:
    """Evaluate Jacobi series at points using forward recurrence.

    Parameters
    ----------
    c : JacobiPolynomialP
        Jacobi series with coefficients shape (...batch, N) and parameters α, β.
    x : Tensor
        Evaluation points, shape (...x_batch).

    Returns
    -------
    Tensor
        Values c(x), shape is broadcast of c's batch dims with x's shape.

    Warnings
    --------
    UserWarning
        If any evaluation points are outside the natural domain [-1, 1].

    Notes
    -----
    Uses forward recurrence to compute the sum f(x) = sum_{k=0}^{n-1} c_k * P_k^{(α,β)}(x).

    The Jacobi polynomials satisfy the three-term recurrence:
        P_0^{(α,β)}(x) = 1
        P_1^{(α,β)}(x) = (α - β)/2 + (α + β + 2)/2 * x

        For n >= 1:
        P_{n+1}^{(α,β)}(x) = ((b_n + c_n*x) * P_n^{(α,β)}(x) - d_n * P_{n-1}^{(α,β)}(x)) / a_n

        where:
        a_n = 2(n+1)(n+α+β+1)(2n+α+β)
        b_n = (2n+α+β+1)(α²-β²)
        c_n = (2n+α+β)(2n+α+β+1)(2n+α+β+2)
        d_n = 2(n+α)(n+β)(2n+α+β+2)

    Examples
    --------
    >>> c = jacobi_polynomial_p(torch.tensor([1.0, 2.0]), alpha=0.5, beta=0.5)
    >>> jacobi_polynomial_p_evaluate(c, torch.tensor([0.0]))
    tensor([1.])
    """
    # Domain check only applies to real tensors (complex roots are expected)
    if not x.is_complex():
        domain = JacobiPolynomialP.DOMAIN

        if ((x < domain[0]) | (x > domain[1])).any():
            warnings.warn(
                f"Evaluating JacobiPolynomialP outside natural domain "
                f"[{domain[0]}, {domain[1]}]. Results may be numerically unstable.",
                stacklevel=2,
            )

    coeffs = c.coeffs
    alpha = c.alpha
    beta = c.beta
    n = coeffs.shape[-1]

    # Handle trivial cases
    if n == 0:
        return x * 0.0
    if n == 1:
        return (
            coeffs[..., 0].expand_as(x).clone()
            if coeffs.dim() == 1
            else coeffs[..., 0:1] * (x * 0.0 + 1.0)
        )

    ab = alpha + beta

    # Use forward recurrence to compute P_k(x) values and accumulate sum
    # Non-batched case (1D coeffs)
    if coeffs.dim() == 1:
        # P_0 = 1
        P_prev_prev = torch.ones_like(x)
        result = coeffs[0] * P_prev_prev

        # P_1 = (α - β)/2 + (α + β + 2)/2 * x
        P_prev = (alpha - beta) / 2 + (ab + 2) / 2 * x
        result = result + coeffs[1] * P_prev

        if n == 2:
            return result

        # Recurrence for P_k, k >= 2
        for k in range(1, n - 1):
            k_f = float(k)
            two_k_ab = 2 * k_f + ab

            # Recurrence coefficients
            a_k = 2 * (k_f + 1) * (k_f + ab + 1) * two_k_ab
            b_k = (two_k_ab + 1) * (alpha * alpha - beta * beta)
            c_k = two_k_ab * (two_k_ab + 1) * (two_k_ab + 2)
            d_k = 2 * (k_f + alpha) * (k_f + beta) * (two_k_ab + 2)

            # P_{k+1} = ((b_k + c_k*x) * P_k - d_k * P_{k-1}) / a_k
            P_curr = ((b_k + c_k * x) * P_prev - d_k * P_prev_prev) / a_k

            result = result + coeffs[k + 1] * P_curr
            P_prev_prev = P_prev
            P_prev = P_curr

        return result

    # Batched case: coeffs shape (...batch, N), x shape (...x_batch)
    batch_shape = coeffs.shape[:-1]

    # Expand coeffs for broadcasting: (...batch, 1, 1, ..., N)
    coeffs_expanded = coeffs
    for _ in range(x.dim()):
        coeffs_expanded = coeffs_expanded.unsqueeze(-2)

    # Expand x for broadcasting: (1, 1, ..., ...x_batch)
    x_expanded = x
    for _ in range(len(batch_shape)):
        x_expanded = x_expanded.unsqueeze(0)

    # P_0 = 1
    P_prev_prev = torch.ones_like(x_expanded)
    result = coeffs_expanded[..., 0] * P_prev_prev

    # P_1 = (α - β)/2 + (α + β + 2)/2 * x
    P_prev = (alpha - beta) / 2 + (ab + 2) / 2 * x_expanded
    result = result + coeffs_expanded[..., 1] * P_prev

    if n == 2:
        return result

    # Recurrence for P_k, k >= 2
    for k in range(1, n - 1):
        k_f = float(k)
        two_k_ab = 2 * k_f + ab

        a_k = 2 * (k_f + 1) * (k_f + ab + 1) * two_k_ab
        b_k = (two_k_ab + 1) * (alpha * alpha - beta * beta)
        c_k = two_k_ab * (two_k_ab + 1) * (two_k_ab + 2)
        d_k = 2 * (k_f + alpha) * (k_f + beta) * (two_k_ab + 2)

        P_curr = ((b_k + c_k * x_expanded) * P_prev - d_k * P_prev_prev) / a_k

        result = result + coeffs_expanded[..., k + 1] * P_curr
        P_prev_prev = P_prev
        P_prev = P_curr

    return result
