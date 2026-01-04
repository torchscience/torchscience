"""Tests for bilinear transform functions."""

import pytest
import torch
from scipy import signal as scipy_signal

from torchscience.signal_processing.filter_design import bilinear_zpk, buttap


class TestBilinearZpk:
    """Tests for bilinear_zpk (analog to digital transform)."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("fs", [1.0, 2.0, 8000.0])
    def test_matches_scipy(self, n: int, fs: float) -> None:
        """Should match scipy.signal.bilinear_zpk."""
        z, p, k = buttap(n, dtype=torch.float64)

        z_d, p_d, k_d = bilinear_zpk(z, p, k, fs=fs)

        # Scipy reference
        z_sp, p_sp, k_sp = scipy_signal.bilinear_zpk(
            z.numpy(), p.numpy(), k.item(), fs=fs
        )

        # Digital filter has same number of poles
        assert p_d.numel() == n

        # Compare poles (sort for consistent ordering)
        p_d_sorted = sorted(p_d.numpy(), key=lambda x: (x.real, x.imag))
        p_sp_sorted = sorted(p_sp, key=lambda x: (x.real, x.imag))

        for p_ts, p_ref in zip(p_d_sorted, p_sp_sorted):
            assert abs(p_ts - p_ref) < 1e-10, (
                f"Pole mismatch: {p_ts} vs {p_ref}"
            )

        # Compare gain
        assert abs(k_d.item() - k_sp) < 1e-10

    def test_poles_inside_unit_circle(self) -> None:
        """Digital filter poles should be inside unit circle (stable)."""
        z, p, k = buttap(4, dtype=torch.float64)
        z_d, p_d, k_d = bilinear_zpk(z, p, k, fs=2.0)

        for pole in p_d:
            assert abs(pole) < 1.0 + 1e-10, f"Pole {pole} outside unit circle"

    def test_adds_zeros_at_nyquist(self) -> None:
        """Bilinear transform adds zeros at z=-1 (Nyquist)."""
        z, p, k = buttap(4, dtype=torch.float64)
        z_d, p_d, k_d = bilinear_zpk(z, p, k, fs=2.0)

        # Should have n zeros at z=-1
        assert z_d.numel() == p_d.numel()
        for zero in z_d:
            assert abs(zero + 1.0) < 1e-10, f"Zero {zero} not at -1"


class TestBilinearZpkGradients:
    """Test gradients for bilinear_zpk."""

    def test_gradient_wrt_poles(self) -> None:
        """Should have gradient w.r.t. analog poles."""
        # Create simple analog filter with differentiable poles
        p = torch.tensor(
            [-1.0 + 0.5j, -1.0 - 0.5j],
            dtype=torch.complex128,
            requires_grad=True,
        )
        z = torch.empty(0, dtype=p.dtype)
        k = torch.tensor(1.0, dtype=torch.float64)

        z_d, p_d, k_d = bilinear_zpk(z, p, k, fs=2.0)

        # Compute loss and backward
        loss = p_d.abs().sum()
        loss.backward()

        assert p.grad is not None
        assert not torch.isnan(p.grad).any()
