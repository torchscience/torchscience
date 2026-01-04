from __future__ import annotations

import torch

from ._polynomial import Polynomial


def polynomial_subtract(p: Polynomial, q: Polynomial) -> Polynomial:
    """Subtract q from p.

    Broadcasts batch dimensions. Result degree is max(deg(p), deg(q)).

    Parameters
    ----------
    p, q : Polynomial
        Polynomials.

    Returns
    -------
    Polynomial
        Difference p - q.
    """
    p_coeffs = p.coeffs
    q_coeffs = q.coeffs

    # Pad shorter polynomial with zeros
    n_p = p_coeffs.shape[-1]
    n_q = q_coeffs.shape[-1]

    if n_p < n_q:
        padding = [0] * (2 * (p_coeffs.dim() - 1)) + [0, n_q - n_p]
        p_coeffs = torch.nn.functional.pad(p_coeffs, padding)
    elif n_q < n_p:
        padding = [0] * (2 * (q_coeffs.dim() - 1)) + [0, n_p - n_q]
        q_coeffs = torch.nn.functional.pad(q_coeffs, padding)

    return Polynomial(coeffs=p_coeffs - q_coeffs)
