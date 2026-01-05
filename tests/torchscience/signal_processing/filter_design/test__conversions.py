"""Tests for filter representation conversions."""

import pytest
import torch
from scipy import signal as scipy_signal

from torchscience.signal_processing.filter_design import (
    bilinear_zpk,
    buttap,
    zpk2sos,
)


class TestZpk2sos:
    """Tests for zpk2sos (zeros-poles-gain to second-order sections)."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_matches_scipy_digital(self, n: int) -> None:
        """Should produce equivalent filter to scipy.signal.zpk2sos."""
        # Create digital Butterworth filter
        z, p, k = buttap(n, dtype=torch.float64)
        z_d, p_d, k_d = bilinear_zpk(z, p, k, fs=2.0)

        sos = zpk2sos(z_d, p_d, k_d)

        # Scipy reference
        sos_scipy = scipy_signal.zpk2sos(z_d.numpy(), p_d.numpy(), k_d.item())

        # Check shape
        assert sos.shape == torch.Size(sos_scipy.shape)

        # Compare frequency response (more robust than coefficient comparison)
        w = torch.linspace(0, 3.14159, 100)
        h_ts = _sos_freqz(sos, w)
        h_scipy = _sos_freqz(torch.from_numpy(sos_scipy), w)

        torch.testing.assert_close(
            h_ts.abs(), h_scipy.abs(), rtol=1e-5, atol=1e-10
        )

    def test_output_shape(self) -> None:
        """SOS should have shape (n_sections, 6)."""
        z, p, k = buttap(4, dtype=torch.float64)
        z_d, p_d, k_d = bilinear_zpk(z, p, k, fs=2.0)

        sos = zpk2sos(z_d, p_d, k_d)

        # 4 poles -> 2 second-order sections
        assert sos.shape == (2, 6)

    def test_odd_order(self) -> None:
        """Odd order should include a first-order section."""
        z, p, k = buttap(5, dtype=torch.float64)
        z_d, p_d, k_d = bilinear_zpk(z, p, k, fs=2.0)

        sos = zpk2sos(z_d, p_d, k_d)

        # 5 poles -> 3 sections (2 second-order + 1 first-order)
        assert sos.shape == (3, 6)


def _sos_freqz(sos: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compute frequency response of SOS filter."""
    h = torch.ones(len(w), dtype=torch.complex128)
    z = torch.exp(1j * w)

    for section in sos:
        b0, b1, b2, a0, a1, a2 = section.to(torch.float64)
        num = b0 + b1 * z**-1 + b2 * z**-2
        den = a0 + a1 * z**-1 + a2 * z**-2
        h = h * num / den

    return h
