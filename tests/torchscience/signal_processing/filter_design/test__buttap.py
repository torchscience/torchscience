"""Tests for Butterworth analog lowpass prototype."""

import pytest
import torch
from scipy import signal as scipy_signal

from torchscience.signal_processing.filter_design import buttap


class TestButtapForward:
    """Test buttap forward correctness."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_poles_match_scipy(self, n: int) -> None:
        """Poles should match scipy.signal.buttap."""
        z, p, k = buttap(n)

        # Get scipy reference
        z_scipy, p_scipy, k_scipy = scipy_signal.buttap(n)

        # Check zeros (should be empty for Butterworth)
        assert z.numel() == 0
        assert len(z_scipy) == 0

        # Check poles (sort by angle for comparison)
        p_np = p.numpy()
        p_sorted = sorted(p_np, key=lambda x: (x.real, x.imag))
        p_scipy_sorted = sorted(p_scipy, key=lambda x: (x.real, x.imag))

        for p_ts, p_sp in zip(p_sorted, p_scipy_sorted):
            assert abs(p_ts - p_sp) < 1e-10, f"Pole mismatch: {p_ts} vs {p_sp}"

        # Check gain
        assert abs(k.item() - k_scipy) < 1e-10

    def test_order_1_single_real_pole(self) -> None:
        """Order 1 should have single real pole at -1."""
        z, p, k = buttap(1)

        assert z.numel() == 0
        assert p.numel() == 1
        assert abs(p[0].real + 1.0) < 1e-10
        assert abs(p[0].imag) < 1e-10
        assert abs(k.item() - 1.0) < 1e-10

    def test_poles_on_unit_circle(self) -> None:
        """All poles should lie on the unit circle."""
        for n in range(1, 9):
            z, p, k = buttap(n)
            for pole in p:
                magnitude = abs(pole)
                assert abs(magnitude - 1.0) < 1e-10, (
                    f"Pole {pole} not on unit circle"
                )

    def test_poles_in_left_half_plane(self) -> None:
        """All poles should be in the left half-plane (stable)."""
        for n in range(1, 9):
            z, p, k = buttap(n)
            for pole in p:
                assert pole.real < 1e-10, f"Pole {pole} not in left half-plane"


class TestButtapDtypes:
    """Test buttap dtype handling."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_output_dtype(self, dtype: torch.dtype) -> None:
        """Output should match requested dtype."""
        z, p, k = buttap(4, dtype=dtype)

        # Poles are complex, so check the underlying real dtype
        if dtype == torch.float32:
            assert p.dtype == torch.complex64
        else:
            assert p.dtype == torch.complex128
        assert k.dtype == dtype


class TestButtapDevice:
    """Test buttap device handling."""

    def test_cpu_device(self) -> None:
        """Should work on CPU."""
        z, p, k = buttap(4, device=torch.device("cpu"))
        assert p.device.type == "cpu"
        assert k.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_device(self) -> None:
        """Should work on CUDA."""
        z, p, k = buttap(4, device=torch.device("cuda"))
        assert p.device.type == "cuda"
        assert k.device.type == "cuda"


class TestButtapEdgeCases:
    """Test buttap edge cases."""

    def test_invalid_order_zero(self) -> None:
        """Order 0 should raise error."""
        with pytest.raises((ValueError, RuntimeError)):
            buttap(0)

    def test_invalid_order_negative(self) -> None:
        """Negative order should raise error."""
        with pytest.raises((ValueError, RuntimeError)):
            buttap(-1)
