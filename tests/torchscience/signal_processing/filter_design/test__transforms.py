"""Tests for frequency transform functions."""

import pytest
import torch
from scipy import signal as scipy_signal

from torchscience.signal_processing.filter_design import (
    buttap,
    lp2bp_zpk,
    lp2hp_zpk,
    lp2lp_zpk,
)


class TestLp2lpZpk:
    """Tests for lp2lp_zpk (lowpass to lowpass frequency scaling)."""

    def test_identity_transform(self) -> None:
        """wo=1.0 should not change the filter."""
        z, p, k = buttap(4, dtype=torch.float64)
        z2, p2, k2 = lp2lp_zpk(z, p, k, wo=1.0)

        torch.testing.assert_close(p2, p)
        torch.testing.assert_close(k2, k)

    def test_frequency_scaling(self) -> None:
        """Poles should scale by wo."""
        z, p, k = buttap(4, dtype=torch.float64)
        wo = 2.0
        z2, p2, k2 = lp2lp_zpk(z, p, k, wo=wo)

        # Poles should be scaled by wo
        torch.testing.assert_close(p2, p * wo)

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("wo", [0.5, 1.0, 2.0, 10.0])
    def test_matches_scipy(self, n: int, wo: float) -> None:
        """Should match scipy.signal.lp2lp_zpk."""
        z, p, k = buttap(n, dtype=torch.float64)

        z2, p2, k2 = lp2lp_zpk(z, p, k, wo=wo)

        # Scipy reference
        z_sp, p_sp, k_sp = scipy_signal.lp2lp_zpk(
            z.numpy(), p.numpy(), k.item(), wo=wo
        )

        # Compare (sort poles for consistent ordering)
        p2_sorted = sorted(p2.numpy(), key=lambda x: (x.real, x.imag))
        p_sp_sorted = sorted(p_sp, key=lambda x: (x.real, x.imag))

        for p_ts, p_ref in zip(p2_sorted, p_sp_sorted):
            assert abs(p_ts - p_ref) < 1e-10

        assert abs(k2.item() - k_sp) < 1e-10


class TestLp2lpZpkGradients:
    """Test gradients for lp2lp_zpk."""

    def test_gradient_wrt_wo(self) -> None:
        """Should have gradient w.r.t. wo."""
        z, p, k = buttap(4, dtype=torch.float64)

        wo = torch.tensor(2.0, requires_grad=True)
        z2, p2, k2 = lp2lp_zpk(z, p, k, wo=wo)

        # Compute loss and backward
        loss = p2.abs().sum() + k2
        loss.backward()

        assert wo.grad is not None
        assert not torch.isnan(wo.grad)


class TestLp2hpZpk:
    """Tests for lp2hp_zpk (lowpass to highpass transform)."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("wo", [0.5, 1.0, 2.0, 10.0])
    def test_matches_scipy(self, n: int, wo: float) -> None:
        """Should match scipy.signal.lp2hp_zpk."""
        z, p, k = buttap(n, dtype=torch.float64)

        z2, p2, k2 = lp2hp_zpk(z, p, k, wo=wo)

        # Scipy reference
        z_sp, p_sp, k_sp = scipy_signal.lp2hp_zpk(
            z.numpy(), p.numpy(), k.item(), wo=wo
        )

        # Compare zeros (highpass has zeros at origin)
        assert z2.numel() == len(z_sp)

        # Compare poles (sort for consistent ordering)
        p2_sorted = sorted(p2.numpy(), key=lambda x: (x.real, x.imag))
        p_sp_sorted = sorted(p_sp, key=lambda x: (x.real, x.imag))

        for p_ts, p_ref in zip(p2_sorted, p_sp_sorted):
            assert abs(p_ts - p_ref) < 1e-10

        assert abs(k2.item() - k_sp) < 1e-10

    def test_adds_zeros_at_origin(self) -> None:
        """Highpass transform adds zeros at s=0."""
        z, p, k = buttap(4, dtype=torch.float64)
        z2, p2, k2 = lp2hp_zpk(z, p, k, wo=1.0)

        # Should have n zeros at origin (degree difference)
        assert z2.numel() == p.numel()
        for zero in z2:
            assert abs(zero) < 1e-10


class TestLp2hpZpkGradients:
    """Test gradients for lp2hp_zpk."""

    def test_gradient_wrt_wo(self) -> None:
        """Should have gradient w.r.t. wo."""
        z, p, k = buttap(4, dtype=torch.float64)

        wo = torch.tensor(2.0, requires_grad=True)
        z2, p2, k2 = lp2hp_zpk(z, p, k, wo=wo)

        # Compute loss and backward
        loss = p2.abs().sum() + k2
        loss.backward()

        assert wo.grad is not None
        assert not torch.isnan(wo.grad)


class TestLp2bpZpk:
    """Tests for lp2bp_zpk (lowpass to bandpass transform)."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    @pytest.mark.parametrize("wo", [1.0, 2.0, 10.0])
    @pytest.mark.parametrize("bw", [0.5, 1.0, 2.0])
    def test_matches_scipy(self, n: int, wo: float, bw: float) -> None:
        """Should match scipy.signal.lp2bp_zpk."""
        z, p, k = buttap(n, dtype=torch.float64)

        z2, p2, k2 = lp2bp_zpk(z, p, k, wo=wo, bw=bw)

        # Scipy reference
        z_sp, p_sp, k_sp = scipy_signal.lp2bp_zpk(
            z.numpy(), p.numpy(), k.item(), wo=wo, bw=bw
        )

        # Bandpass doubles the order
        assert p2.numel() == 2 * n
        assert len(p_sp) == 2 * n

        # Compare poles (sort for consistent ordering)
        p2_sorted = sorted(p2.numpy(), key=lambda x: (x.real, x.imag))
        p_sp_sorted = sorted(p_sp, key=lambda x: (x.real, x.imag))

        for p_ts, p_ref in zip(p2_sorted, p_sp_sorted):
            assert abs(p_ts - p_ref) < 1e-10, (
                f"Pole mismatch: {p_ts} vs {p_ref}"
            )

        assert abs(k2.item() - k_sp) < 1e-10

    def test_doubles_filter_order(self) -> None:
        """Bandpass transform doubles the number of poles."""
        z, p, k = buttap(4, dtype=torch.float64)
        z2, p2, k2 = lp2bp_zpk(z, p, k, wo=1.0, bw=0.5)

        assert p2.numel() == 2 * p.numel()

    def test_adds_zeros_at_origin(self) -> None:
        """Bandpass transform adds zeros at s=0."""
        z, p, k = buttap(4, dtype=torch.float64)
        z2, p2, k2 = lp2bp_zpk(z, p, k, wo=1.0, bw=0.5)

        # Should have n zeros at origin
        assert z2.numel() == p.numel()
        for zero in z2:
            assert abs(zero) < 1e-10


class TestLp2bpZpkGradients:
    """Test gradients for lp2bp_zpk."""

    def test_gradient_wrt_wo_and_bw(self) -> None:
        """Should have gradient w.r.t. wo and bw."""
        z, p, k = buttap(4, dtype=torch.float64)

        wo = torch.tensor(2.0, requires_grad=True)
        bw = torch.tensor(0.5, requires_grad=True)
        z2, p2, k2 = lp2bp_zpk(z, p, k, wo=wo, bw=bw)

        # Compute loss and backward
        loss = p2.abs().sum() + k2
        loss.backward()

        assert wo.grad is not None
        assert bw.grad is not None
        assert not torch.isnan(wo.grad)
        assert not torch.isnan(bw.grad)
