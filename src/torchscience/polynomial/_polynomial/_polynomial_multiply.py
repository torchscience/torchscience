from __future__ import annotations

import torch

from ._polynomial import Polynomial


def polynomial_multiply(p: Polynomial, q: Polynomial) -> Polynomial:
    """Multiply two polynomials.

    Computes convolution of coefficients. Result degree is deg(p) + deg(q).

    Parameters
    ----------
    p, q : Polynomial
        Polynomials to multiply.

    Returns
    -------
    Polynomial
        Product p * q.
    """
    p_coeffs = p.coeffs
    q_coeffs = q.coeffs

    # Handle batch dimensions
    # We need to broadcast batch dimensions and convolve coefficient dimension

    # Get shapes
    p_batch = p_coeffs.shape[:-1]
    q_batch = q_coeffs.shape[:-1]
    n_p = p_coeffs.shape[-1]
    n_q = q_coeffs.shape[-1]

    # Broadcast batch dimensions
    broadcast_batch = torch.broadcast_shapes(p_batch, q_batch)

    # Expand to broadcast shape
    p_expanded = p_coeffs.expand(*broadcast_batch, n_p)
    q_expanded = q_coeffs.expand(*broadcast_batch, n_q)

    # For non-batched case, use explicit convolution
    if len(broadcast_batch) == 0:
        # Simple 1D convolution
        n_out = n_p + n_q - 1
        result = torch.zeros(
            n_out, dtype=p_coeffs.dtype, device=p_coeffs.device
        )
        for i in range(n_p):
            for j in range(n_q):
                result[i + j] = result[i + j] + p_expanded[i] * q_expanded[j]
        return Polynomial(coeffs=result)

    # For batched case, use explicit convolution per batch element
    # Flatten batch dimensions
    batch_size = broadcast_batch.numel() if len(broadcast_batch) > 0 else 1
    p_flat = p_expanded.reshape(batch_size, n_p)
    q_flat = q_expanded.reshape(batch_size, n_q)

    n_out = n_p + n_q - 1
    result = torch.zeros(
        batch_size, n_out, dtype=p_coeffs.dtype, device=p_coeffs.device
    )

    for i in range(n_p):
        for j in range(n_q):
            result[:, i + j] = result[:, i + j] + p_flat[:, i] * q_flat[:, j]

    # Reshape back to broadcast batch dimensions
    result = result.reshape(*broadcast_batch, n_out)
    return Polynomial(coeffs=result)
