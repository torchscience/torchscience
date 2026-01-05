"""Unified Butterworth filter design function."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from ._bilinear import bilinear_zpk
from ._buttap import buttap
from ._conversions import zpk2sos
from ._transforms import lp2bp_zpk, lp2bs_zpk, lp2hp_zpk, lp2lp_zpk


def butterworth(
    n: int,
    cutoff: Tensor | float | list[float],
    btype: Literal["lowpass", "highpass", "bandpass", "bandstop"] = "lowpass",
    output: Literal["sos", "zpk", "ba"] = "sos",
    fs: float | None = None,
) -> Tensor | tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor]:
    """Design an Nth-order digital Butterworth filter.

    Parameters
    ----------
    n : int
        The order of the filter.
    cutoff : Tensor or float or list[float]
        The critical frequency or frequencies. For lowpass and highpass, this
        is a scalar. For bandpass and bandstop, this is a length-2 sequence
        [low, high]. Frequencies are expressed as a fraction of the Nyquist
        frequency (0 to 1), unless fs is specified.
    btype : {"lowpass", "highpass", "bandpass", "bandstop"}, optional
        The type of filter. Default is "lowpass".
    output : {"sos", "zpk", "ba"}, optional
        Type of output:
        - "sos": second-order sections (default, recommended)
        - "zpk": zeros, poles, gain
        - "ba": numerator/denominator
    fs : float, optional
        The sampling frequency of the digital system. If specified, cutoff
        is in the same units as fs (e.g., Hz).

    Returns
    -------
    sos : Tensor
        Second-order sections representation of the filter (if output="sos").
        Shape: (n_sections, 6) where each row is [b0, b1, b2, a0, a1, a2].
    z, p, k : tuple of Tensors
        Zeros, poles, and gain of the filter (if output="zpk").
    b, a : tuple of Tensors
        Numerator and denominator of the filter (if output="ba").

    Notes
    -----
    The Butterworth filter has maximally flat frequency response in the
    passband. The filter is designed by:
    1. Creating an analog Butterworth lowpass prototype
    2. Transforming to the desired frequency band
    3. Converting to digital using the bilinear transform

    Examples
    --------
    >>> import torch
    >>> from torchscience.signal_processing.filter_design import butterworth
    >>> sos = butterworth(4, 0.3)  # 4th order lowpass at 0.3 * Nyquist
    >>> sos.shape
    torch.Size([2, 6])

    >>> z, p, k = butterworth(4, 0.3, output="zpk")
    >>> z.shape, p.shape
    (torch.Size([4]), torch.Size([4]))

    >>> sos = butterworth(4, 100.0, fs=1000.0)  # 100 Hz cutoff at 1000 Hz sample rate
    """
    # Convert cutoff to tensor
    if isinstance(cutoff, (int, float)):
        cutoff_tensor = torch.tensor(cutoff, dtype=torch.float64)
    elif isinstance(cutoff, list):
        cutoff_tensor = torch.tensor(cutoff, dtype=torch.float64)
    else:
        cutoff_tensor = cutoff.to(torch.float64)

    # Normalize by sampling frequency if provided
    if fs is not None:
        nyquist = fs / 2.0
        cutoff_tensor = cutoff_tensor / nyquist

    # Validate cutoff range (must be 0 < cutoff < 1 for digital)
    if cutoff_tensor.numel() == 1:
        if not (0 < cutoff_tensor.item() < 1):
            raise ValueError(
                f"Cutoff frequency must be between 0 and 1 (Nyquist), got {cutoff_tensor.item()}"
            )
    else:
        if not (0 < cutoff_tensor[0].item() < cutoff_tensor[1].item() < 1):
            raise ValueError(
                f"Cutoff frequencies must satisfy 0 < low < high < 1, got {cutoff_tensor.tolist()}"
            )

    # Get analog Butterworth lowpass prototype
    z_analog, p_analog, k_analog = buttap(n)

    # Pre-warp the cutoff frequencies for bilinear transform
    # With fs=2 for normalized frequency: w = 2 * fs * tan(pi * f / fs) = 4 * tan(pi * f / 2)
    warped = 4.0 * torch.tan(torch.pi * cutoff_tensor / 2.0)

    # Apply frequency transformation
    if btype == "lowpass":
        z_transformed, p_transformed, k_transformed = lp2lp_zpk(
            z_analog, p_analog, k_analog, warped
        )
    elif btype == "highpass":
        z_transformed, p_transformed, k_transformed = lp2hp_zpk(
            z_analog, p_analog, k_analog, warped
        )
    elif btype == "bandpass":
        bw = warped[1] - warped[0]
        w0 = torch.sqrt(warped[0] * warped[1])
        z_transformed, p_transformed, k_transformed = lp2bp_zpk(
            z_analog, p_analog, k_analog, w0, bw
        )
    elif btype == "bandstop":
        bw = warped[1] - warped[0]
        w0 = torch.sqrt(warped[0] * warped[1])
        z_transformed, p_transformed, k_transformed = lp2bs_zpk(
            z_analog, p_analog, k_analog, w0, bw
        )
    else:
        raise ValueError(f"Invalid btype: {btype}")

    # Bilinear transform to digital (fs=2 for normalized frequency)
    z_digital, p_digital, k_digital = bilinear_zpk(
        z_transformed, p_transformed, k_transformed, fs=2.0
    )

    # Return in requested format
    if output == "zpk":
        return z_digital, p_digital, k_digital
    elif output == "sos":
        return zpk2sos(z_digital, p_digital, k_digital)
    elif output == "ba":
        return _zpk2ba(z_digital, p_digital, k_digital)
    else:
        raise ValueError(f"Invalid output format: {output}")


def _zpk2ba(z: Tensor, p: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
    """Convert zeros, poles, gain to transfer function coefficients.

    Parameters
    ----------
    z : Tensor
        Zeros of the transfer function.
    p : Tensor
        Poles of the transfer function.
    k : Tensor
        Gain of the transfer function.

    Returns
    -------
    b : Tensor
        Numerator polynomial coefficients.
    a : Tensor
        Denominator polynomial coefficients.
    """
    # Build polynomials from roots
    b = k * _poly_from_roots(z)
    a = _poly_from_roots(p)

    # Return real parts (imaginary should be negligible for conjugate pairs)
    return b.real, a.real


def _poly_from_roots(roots: Tensor) -> Tensor:
    """Build polynomial coefficients from roots.

    Computes coefficients of: (x - r0)(x - r1)...(x - rn)

    Parameters
    ----------
    roots : Tensor
        Roots of the polynomial.

    Returns
    -------
    coeffs : Tensor
        Polynomial coefficients in descending order [x^n, x^(n-1), ..., x^0].
    """
    if roots.numel() == 0:
        return torch.ones(1, dtype=roots.dtype, device=roots.device)

    # Start with (x - r0)
    coeffs = torch.tensor(
        [1.0, -roots[0]], dtype=torch.complex128, device=roots.device
    )

    # Multiply by each (x - ri)
    for r in roots[1:]:
        # Convolve [1, -r] with current coefficients
        new_coeffs = torch.zeros(
            len(coeffs) + 1, dtype=torch.complex128, device=roots.device
        )
        for i, c in enumerate(coeffs):
            new_coeffs[i] += c
            new_coeffs[i + 1] -= c * r
        coeffs = new_coeffs

    return coeffs
