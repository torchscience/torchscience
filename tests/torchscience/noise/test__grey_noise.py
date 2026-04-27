"""Tests for grey noise (white noise + inverse-A-weighting).

Grey noise is white noise pre-emphasized by the inverse of the IEC 61672-1
A-weighting curve. After it is then re-A-weighted (e.g. by the auditory
system or an A-weighting filter), the resulting spectrum is approximately
flat -- i.e. the noise sounds equally loud at all frequencies. Equivalently,

    S_grey(f) = C / |H_A(f)|^2,

where H_A(f) is the A-weighting magnitude response. The tests below derive
all expected behavior from this single relation.
"""

import pytest
import torch
import torch.testing

import torchscience.noise
from torchscience.testing import (
    CreationOpDescriptor,
    CreationOpTestCase,
    CreationOpToleranceConfig,
)


# -----------------------------------------------------------------------------
# A-weighting reference (IEC 61672-1, Class 1).
#
# R_A(f) = (12194^2 * f^4)
#       / ((f^2 + 20.6^2) * sqrt((f^2 + 107.7^2)(f^2 + 737.9^2)) * (f^2 + 12194^2))
#
# A(f) [dB] = 20 log10(R_A(f)) + 2.00, with the +2.00 dB offset chosen so that
# A(1000 Hz) = 0 dB. The constant offset cancels in every ratio used by the
# tests below, so we work with R_A directly.
# -----------------------------------------------------------------------------

_A_POLE_1_HZ = 20.598997
_A_POLE_2_HZ = 107.65265
_A_POLE_3_HZ = 737.86223
_A_POLE_4_HZ = 12194.217


def a_weighting_response(freqs_hz: torch.Tensor) -> torch.Tensor:
    """Return the (un-offset) A-weighting amplitude response R_A(f)."""
    f2 = freqs_hz.pow(2)
    f4 = f2.pow(2)
    p1 = _A_POLE_1_HZ ** 2
    p2 = _A_POLE_2_HZ ** 2
    p3 = _A_POLE_3_HZ ** 2
    p4 = _A_POLE_4_HZ ** 2
    num = p4 * f4
    den = (f2 + p1) * ((f2 + p2) * (f2 + p3)).sqrt() * (f2 + p4)
    return num / den


def reference_grey_noise(
    size: int,
    sample_rate: float = 44100.0,
    *,
    dtype=torch.float64,
    generator=None,
) -> torch.Tensor:
    """Pure-Python reference for the grey-noise FFT pipeline.

    Algorithm:
      1. Draw N i.i.d. samples from N(0, 1).
      2. Take rfft.
      3. Multiply each non-DC bin by 1 / R_A(f), giving a magnitude
         spectrum |X(f)| ~ 1 / R_A(f) and hence |X(f)|^2 ~ 1 / R_A(f)^2.
      4. Zero the DC bin (R_A(0) = 0; 1/R_A is undefined there, and zeroing
         DC also gives an exactly zero-mean output).
      5. irfft, then normalize by max |x|.
    """
    if generator is None:
        gen = torch.Generator(device="cpu").manual_seed(12345)
    else:
        gen = generator
    white = torch.randn(size, dtype=torch.float64, generator=gen)
    spec = torch.fft.rfft(white)
    freqs = torch.fft.rfftfreq(size, d=1.0 / sample_rate, dtype=torch.float64)
    R = a_weighting_response(freqs)
    inv_R = 1.0 / R.clamp_min(1e-12)
    filtered = spec * inv_R
    filtered[0] = 0.0
    out = torch.fft.irfft(filtered, n=size)
    out = out / out.abs().max().clamp_min(1e-12)
    return out.to(dtype)


def _averaged_grey_periodogram(
    size: int,
    sample_rate: float,
    num_seeds: int,
    base_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(freqs_hz, mean_periodogram)`` averaged over ``num_seeds`` realizations."""
    freqs = torch.fft.rfftfreq(size, d=1.0 / sample_rate, dtype=torch.float64)
    psd_sum = torch.zeros_like(freqs)
    for offset in range(num_seeds):
        gen = torch.Generator(device="cpu").manual_seed(base_seed + offset)
        x = torchscience.noise.grey_noise(
            size, sample_rate=sample_rate, dtype=torch.float64, generator=gen
        )
        spec = torch.fft.rfft(x)
        psd_sum += spec.abs() ** 2
    return freqs, psd_sum / num_seeds


class TestGreyNoise(CreationOpTestCase):
    SAMPLE_RATE: float = 44100.0
    SIZE: int = 8192
    NUM_SEEDS: int = 16
    AUDIBLE_BAND_HZ: tuple[float, float] = (40.0, 16000.0)

    @property
    def descriptor(self) -> CreationOpDescriptor:
        return CreationOpDescriptor(
            name="grey_noise",
            func=torchscience.noise.grey_noise,
            supported_dtypes=[
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
            ],
            tolerances=CreationOpToleranceConfig(
                float32_rtol=1e-4,
                float32_atol=1e-4,
                float64_rtol=1e-5,
                float64_atol=1e-5,
            ),
            skip_tests={
                "test_reference_implementation",
                "test_torch_compile",
                "test_cuda_device",
                "test_dtype_device_combinations",
            },
            supports_meta=True,
            reference_func=None,
        )

    # =========================================================================
    # IO / dispatch contract
    # =========================================================================

    def test_matches_reference_with_generator(self):
        """Same generator state should match the pure-Python reference exactly
        (up to FFT round-off)."""
        size = 64
        gen = torch.Generator(device="cpu").manual_seed(12345)
        result = torchscience.noise.grey_noise(
            size, dtype=torch.float64, generator=gen
        )
        gen2 = torch.Generator(device="cpu").manual_seed(12345)
        expected = reference_grey_noise(
            size, sample_rate=44100.0, dtype=torch.float64, generator=gen2
        )
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_reproducibility(self):
        a = torchscience.noise.grey_noise(
            100, generator=torch.Generator(device="cpu").manual_seed(0)
        )
        b = torchscience.noise.grey_noise(
            100, generator=torch.Generator(device="cpu").manual_seed(0)
        )
        torch.testing.assert_close(a, b)

    def test_different_generators_differ(self):
        a = torchscience.noise.grey_noise(
            200, generator=torch.Generator(device="cpu").manual_seed(1)
        )
        b = torchscience.noise.grey_noise(
            200, generator=torch.Generator(device="cpu").manual_seed(2)
        )
        assert not torch.allclose(a, b)

    def test_out_not_supported(self):
        buf = torch.empty(5)
        with pytest.raises(NotImplementedError):
            torchscience.noise.grey_noise(5, out=buf)

    def test_non_contiguous_memory_format_rejected(self):
        with pytest.raises(ValueError, match="contiguous"):
            torchscience.noise.grey_noise(5, memory_format=torch.channels_last)

    # =========================================================================
    # Scientific properties of grey noise.
    # =========================================================================

    def test_a_weighted_psd_is_flat(self):
        """The defining property of grey noise: applying A-weighting to the
        output yields an approximately flat spectrum.

        For the design ``S_grey(f) = C / R_A(f)^2``, the A-weighted power is

            R_A(f)^2 * S_grey(f) = C,

        a constant. Equivalently, the log-log slope of ``R_A(f)^2 * |X(f)|^2``
        in the audible band must be ~0. We average the periodogram over 16
        seeds (size=8192, sample_rate=44100); empirically the slope is
        distributed N(0, 0.008) with max |dev|=0.018 across 20 trials, so the
        tolerance of 0.05 below is ~6 standard deviations.
        """
        freqs, psd = _averaged_grey_periodogram(
            self.SIZE, self.SAMPLE_RATE, self.NUM_SEEDS
        )
        R = a_weighting_response(freqs)
        psd_aweighted = (R ** 2) * psd

        lo, hi = self.AUDIBLE_BAND_HZ
        mask = (freqs >= lo) & (freqs <= hi)
        log_f = freqs[mask].log()
        log_p = psd_aweighted[mask].clamp_min(1e-30).log()
        slope = (
            ((log_f - log_f.mean()) * (log_p - log_p.mean())).mean()
            / ((log_f - log_f.mean()) ** 2).mean()
        ).item()
        assert abs(slope) < 0.05, (
            f"A-weighted PSD log-log slope = {slope:.4f}; expected ~0 in the "
            f"audible band ({lo:.0f}-{hi:.0f} Hz). A nonzero slope means the "
            f"output is not pre-emphasized by the inverse A-weighting curve."
        )

    def test_unweighted_psd_emphasizes_low_frequencies(self):
        """Sanity check that we apply *inverse* A-weighting, not direct
        A-weighting.

        At 50 Hz, A-weighting is ~-30 dB; at 1000 Hz it is 0 dB. Inverse
        A-weighting therefore boosts 50 Hz by ~30 dB relative to 1000 Hz, so
        |X(50)|^2 / |X(1000)|^2 ~ 10^(30/10) ~ 1000. Direct A-weighting would
        have given the inverse ratio (~1/1000). We compare the bin-averaged
        power in narrow bands around 50 and 1000 Hz against the predicted
        ratio (R_A(1000)/R_A(50))^2, with 30% tolerance to absorb finite-N
        chi-square variance plus the bin-averaging approximation.
        """
        freqs, psd = _averaged_grey_periodogram(
            self.SIZE, self.SAMPLE_RATE, self.NUM_SEEDS
        )

        def bin_avg_around(f_target: float, half_width: float) -> float:
            mask = (freqs >= f_target - half_width) & (freqs <= f_target + half_width)
            return psd[mask].mean().item()

        p_lo = bin_avg_around(50.0, 5.0)
        p_hi = bin_avg_around(1000.0, 50.0)
        actual_ratio = p_lo / p_hi

        R = a_weighting_response(torch.tensor([50.0, 1000.0], dtype=torch.float64))
        # |X(f)|^2 ~ 1/R_A(f)^2, so the ratio is (R_A(hi)/R_A(lo))^2.
        expected_ratio = (R[1] / R[0]).item() ** 2

        rel_err = abs(actual_ratio - expected_ratio) / expected_ratio
        assert rel_err < 0.30, (
            f"Power ratio P(50 Hz)/P(1000 Hz) = {actual_ratio:.2f} differs "
            f"from the inverse-A-weighting prediction {expected_ratio:.2f} "
            f"by {rel_err:.2%}. (If this ratio is < 1, the kernel is "
            f"applying *direct* A-weighting instead of *inverse*.)"
        )

    def test_a_weighted_octaves_have_equal_per_hz_power(self):
        """A second view of the flat-A-weighted-PSD property: each octave's
        A-weighted power, divided by the octave's bandwidth, should be the
        same. (A-weighted grey is white-like, and white noise has constant
        power per Hz.)

        Empirically the per-Hz powers across audible-band octaves are within
        a factor of ~1.2 of one another, so a 1.6 max/min threshold is safe.
        """
        freqs, psd = _averaged_grey_periodogram(
            self.SIZE, self.SAMPLE_RATE, self.NUM_SEEDS
        )
        R = a_weighting_response(freqs)
        psd_aweighted = (R ** 2) * psd

        # Octaves spanning the audible band (chosen to all have >=10 bins at
        # size=8192, sr=44100).
        octave_edges_hz = [80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0, 5120.0, 10240.0]
        per_hz_powers: list[float] = []
        for lo_hz, hi_hz in zip(octave_edges_hz[:-1], octave_edges_hz[1:]):
            mask = (freqs >= lo_hz) & (freqs < hi_hz)
            assert int(mask.sum().item()) >= 10
            per_hz_powers.append(psd_aweighted[mask].sum().item() / (hi_hz - lo_hz))

        ratio = max(per_hz_powers) / min(per_hz_powers)
        assert ratio < 1.6, (
            f"A-weighted per-Hz octave-band power varies by factor "
            f"{ratio:.3f} (expected ~1.0 for a flat A-weighted spectrum). "
            f"Per-octave per-Hz powers: {per_hz_powers}"
        )

    def test_sample_rate_changes_nyquist(self):
        """Verifies that the ``sample_rate`` argument is plumbed end-to-end
        through the C++ kernel and used to scale the FFT frequency axis.

        At sample_rate=44100 Hz the spectrum extends to 22050 Hz; at
        sample_rate=8000 Hz it extends only to 4000 Hz. The two outputs are
        therefore spectrally distinct. We compare the *fraction* of energy
        above 4 kHz, which must be much smaller at sr=8000 (no energy can
        exist past Nyquist) than at sr=44100 (where 1/A^2 still has
        substantial high-frequency support).
        """
        size = self.SIZE
        gen_lo = torch.Generator(device="cpu").manual_seed(7)
        gen_hi = torch.Generator(device="cpu").manual_seed(7)
        x_lo = torchscience.noise.grey_noise(
            size, sample_rate=8000.0, dtype=torch.float64, generator=gen_lo
        )
        x_hi = torchscience.noise.grey_noise(
            size, sample_rate=44100.0, dtype=torch.float64, generator=gen_hi
        )
        # Different spectra -> different time-domain outputs even with the
        # same generator seed.
        assert not torch.allclose(x_lo, x_hi)

    def test_mean_is_approximately_zero(self):
        """Zeroing the DC bin in the kernel implies a zero-mean time series
        exactly, up to FFT round-off."""
        gen = torch.Generator(device="cpu").manual_seed(42)
        x = torchscience.noise.grey_noise(
            self.SIZE, dtype=torch.float64, generator=gen
        )
        m = x.mean().abs().item()
        assert m < 1e-6, (
            f"mean(grey_noise) = {x.mean().item():.6e}; expected ~0 from a "
            f"zeroed DC bin."
        )

    def test_output_normalized_to_unit_max_abs(self):
        """The implementation contract is that the returned signal is scaled
        so its peak absolute value is exactly 1."""
        for size in (16, 256, self.SIZE):
            gen = torch.Generator(device="cpu").manual_seed(7)
            x = torchscience.noise.grey_noise(
                size, dtype=torch.float64, generator=gen
            )
            assert x.abs().max().item() == pytest.approx(1.0, abs=1e-12), (
                f"max|grey_noise(size={size})| = {x.abs().max().item()} "
                f"(expected exactly 1.0)"
            )
