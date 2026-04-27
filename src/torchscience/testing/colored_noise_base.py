"""Reusable scientific-property tests for colored-noise creation operators.

A colored-noise generator produces a 1-D signal whose power spectral density
(PSD) follows a power law,

    S(f) = C * f^alpha,

with the exponent ``alpha`` determining the "color":

    alpha =  0   -> white noise   (flat spectrum)
    alpha = -1   -> pink noise    (1/f, equal energy per octave)
    alpha = -2   -> brown noise   (1/f^2, Brownian / random-walk-like)
    alpha = +1   -> blue noise    (f, opposite of brown)
    alpha = +2   -> violet noise  (f^2, opposite of pink)

Every test in this module is a directly-derivable consequence of the power
law above, parameterized by ``alpha``. Mixing this class into a test class
along with :class:`CreationOpTestCase` therefore gives the same battery of
spectral-shape, integral-structure, and zero-mean / amplitude-contract checks
to every color.
"""

from __future__ import annotations

import math
from abc import abstractmethod

import pytest
import torch


def averaged_periodogram(
    noise_func,
    size: int,
    num_seeds: int,
    base_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(freqs, mean_periodogram)`` averaged over ``num_seeds`` realizations.

    Each FFT bin of a single realization is approximately chi-square distributed
    around the deterministic shaping filter. Averaging ``num_seeds`` independent
    realizations reduces the per-bin variance by a factor of ``num_seeds``, which
    gives stable per-bin power estimates suitable for asserting spectral-shape
    properties without flakiness.
    """
    freqs = torch.fft.rfftfreq(size, d=1.0, dtype=torch.float64)
    psd_sum = torch.zeros_like(freqs)
    for offset in range(num_seeds):
        gen = torch.Generator(device="cpu").manual_seed(base_seed + offset)
        x = noise_func(size, dtype=torch.float64, generator=gen)
        spec = torch.fft.rfft(x)
        psd_sum += spec.abs() ** 2
    return freqs, psd_sum / num_seeds


class ColoredNoiseScientificMixin:
    """Mixin asserting the five scientific properties of a colored-noise op.

    Subclasses must implement :attr:`alpha` (the PSD exponent) and
    :attr:`noise_func` (the operator under test, e.g. ``torchscience.noise.
    pink_noise``). The tests below then derive every threshold directly from
    ``alpha`` rather than baking in color-specific magic numbers.

    The power-law constants used in tests come from integrating
    ``S(f) = C * f^alpha`` over a band ``[a, b]``:

        for alpha != -1:  integral = C * (b^(alpha+1) - a^(alpha+1)) / (alpha+1)
        for alpha == -1:  integral = C * ln(b/a)

    Empirical calibration of the tolerances was done with ``num_seeds=16`` and
    ``size=8192`` across 200 seeds and all five colors. The thresholds below
    are at least ~5 standard deviations away from the worst observed run.
    """

    # Number of independent seeds whose periodograms we average. 16 is enough
    # to reduce per-bin std by a factor of 4, putting the slope estimator in
    # the ~0.005 std regime regardless of alpha.
    NUM_SEEDS: int = 16

    # FFT size used by the spectral-shape tests. Large enough to give us many
    # high-resolution octaves; small enough that the test suite remains fast.
    SIZE: int = 8192

    # Base seed for the averaged-periodogram tests. Reproducible.
    BASE_SEED: int = 0

    @property
    @abstractmethod
    def alpha(self) -> float:
        """The PSD exponent: ``S(f) ~ f^alpha``."""
        ...

    @property
    @abstractmethod
    def noise_func(self):
        """The colored-noise function under test."""
        ...

    # =========================================================================
    # Spectral-shape tests
    # =========================================================================

    def test_log_log_psd_slope_matches_alpha(self):
        """``log S(f) = alpha * log f + const``: the log-log slope equals alpha.

        We average the periodogram over ``NUM_SEEDS`` realizations; empirically
        the resulting slope estimator has std ~0.005 across all five colors,
        so the +/- 0.05 tolerance below is ~10 standard deviations.
        """
        freqs, psd = averaged_periodogram(
            self.noise_func, self.SIZE, self.NUM_SEEDS, self.BASE_SEED
        )
        # Restrict to [4/N, 0.4]: avoid the lowest few bins (where bin
        # discretization dominates) and the Nyquist edge (where finite-N
        # effects bias the estimate).
        mask = (freqs >= 4.0 / self.SIZE) & (freqs <= 0.4)
        log_f = freqs[mask].log()
        log_p = psd[mask].clamp_min(1e-30).log()
        slope = (
            ((log_f - log_f.mean()) * (log_p - log_p.mean())).mean()
            / ((log_f - log_f.mean()) ** 2).mean()
        ).item()
        assert abs(slope - self.alpha) < 0.05, (
            f"log-log PSD slope {slope:.4f} is not consistent with the "
            f"power law f^alpha for alpha={self.alpha} (expected slope "
            f"{self.alpha:.4f} +/- 0.05)"
        )

    def test_octave_power_ratio_matches_power_law(self):
        """Doubling the band shifts the power by ``2^(alpha+1)``.

        For ``S(f) = C * f^alpha`` and any ``f0`` (with alpha != -1),

            integral_{f0}^{2 f0} C f^alpha df
                = C f0^(alpha+1) (2^(alpha+1) - 1) / (alpha+1).

        Doubling ``f0`` multiplies this by ``2^(alpha+1)``. Composed across K
        octaves, the ratio of two octaves K apart is ``2^((alpha+1) K)``. We
        use the widest available lever arm (first vs. last qualifying octave)
        because per-octave chi-square variance is small in absolute terms but
        compounds across narrowly-separated octaves; widely-separated octaves
        give a much higher signal-to-noise ratio (empirically <=18% deviation
        in 100 trials across all colors, 30% tolerance is ~5sigma).

        For alpha == -1 (pink), the predicted ratio is exactly 1 -- this is
        the famous "equal energy per octave" property.
        """
        freqs, psd = averaged_periodogram(
            self.noise_func, self.SIZE, self.NUM_SEEDS, self.BASE_SEED
        )
        # Build octaves [f0, 2*f0) entirely below 0.4 (Nyquist guardrail) with
        # at least 32 bins each (for low chi-square variance per band).
        octaves: list[tuple[float, float]] = []
        cur_lo = 4.0 / self.SIZE
        while cur_lo * 2.0 <= 0.4:
            cur_hi = cur_lo * 2.0
            in_band = (freqs >= cur_lo) & (freqs < cur_hi)
            if int(in_band.sum().item()) >= 32:
                octaves.append((cur_lo, cur_hi))
            cur_lo = cur_hi

        assert len(octaves) >= 4, (
            f"Need at least 4 high-resolution octaves below the Nyquist "
            f"guardrail; got {len(octaves)} (size={self.SIZE})."
        )

        lo_lo, lo_hi = octaves[0]
        hi_lo, hi_hi = octaves[-1]
        p_lo = psd[(freqs >= lo_lo) & (freqs < lo_hi)].sum().item()
        p_hi = psd[(freqs >= hi_lo) & (freqs < hi_hi)].sum().item()

        K = len(octaves) - 1
        expected_ratio = 2.0 ** ((self.alpha + 1.0) * K)
        actual_ratio = p_hi / p_lo
        rel_err = abs(actual_ratio - expected_ratio) / expected_ratio
        assert rel_err < 0.30, (
            f"High/low octave power ratio = {actual_ratio:.4g} differs from "
            f"the f^alpha prediction 2^((alpha+1) K) = {expected_ratio:.4g} "
            f"by {rel_err:.2%} (alpha={self.alpha}, K={K} octaves apart)."
        )

    def test_integral_ratio_matches_power_law(self):
        """A non-octave-aligned form of the power-law check.

        For any two bands ``[a, b]`` and ``[c, d]``, integrating ``C f^alpha``
        gives

            P([a,b])   b^(alpha+1) - a^(alpha+1)
            -------- = -------------------------     (alpha != -1)
            P([c,d])   d^(alpha+1) - c^(alpha+1)

        and the log-ratio limit ``ln(b/a) / ln(d/c)`` for alpha == -1. The
        bands below are intentionally not octave-aligned, so this is an
        independent assertion from the octave test. Empirically the realized
        relative error is at most ~5% across colors with NUM_SEEDS=16; the
        10% tolerance below is ~5 sigma.
        """
        freqs, psd = averaged_periodogram(
            self.noise_func, self.SIZE, self.NUM_SEEDS, self.BASE_SEED
        )
        band_pairs = [
            ((0.01, 0.05), (0.05, 0.25)),
            ((0.005, 0.05), (0.05, 0.5)),
            ((0.01, 0.10), (0.05, 0.25)),
        ]
        for (a, b), (c, d) in band_pairs:
            if self.alpha == -1.0:
                expected = math.log(b / a) / math.log(d / c)
            else:
                p = self.alpha + 1.0
                expected = (b**p - a**p) / (d**p - c**p)

            p1 = psd[(freqs >= a) & (freqs < b)].sum().item()
            p2 = psd[(freqs >= c) & (freqs < d)].sum().item()
            ratio = p1 / p2
            rel_err = abs(ratio - expected) / abs(expected)
            assert rel_err < 0.10, (
                f"Band power ratio P([{a},{b}))/P([{c},{d}))={ratio:.4g} "
                f"deviates by {rel_err:.2%} from the f^{self.alpha} "
                f"prediction {expected:.4g}."
            )

    # =========================================================================
    # Time-domain contract tests
    # =========================================================================

    def test_mean_is_approximately_zero(self):
        """The output is zero-mean (DC bin is zero in the frequency domain).

        For pink/brown the DC bin is undefined under the power law and is
        zeroed by construction; for blue/violet the filter ``f^(alpha/2)``
        already vanishes at f=0; for white the DC bin must also be zeroed
        explicitly so the mean contract is the same across colors.

        Since ``mean(x) = X[0] / N``, a zero DC bin gives a zero-mean time
        series exactly (up to FFT round-off), and max-abs normalization
        preserves this. The 1e-6 threshold is many orders above float64
        round-off but below any meaningful DC leakage.
        """
        gen = torch.Generator(device="cpu").manual_seed(42)
        x = self.noise_func(self.SIZE, dtype=torch.float64, generator=gen)
        m = x.mean().abs().item()
        assert m < 1e-6, (
            f"mean({type(self).__name__} output) = {x.mean().item():.6e}; "
            f"expected ~0 from a zeroed DC bin."
        )

    def test_output_normalized_to_unit_max_abs(self):
        """The output is scaled so its peak absolute value is exactly 1.

        This is required for the spectral tests above to be normalization-
        independent: scaling x -> alpha*x scales the PSD by alpha^2, which
        cancels in every ratio used.
        """
        for size in (16, 256, self.SIZE):
            gen = torch.Generator(device="cpu").manual_seed(7)
            x = self.noise_func(size, dtype=torch.float64, generator=gen)
            assert x.abs().max().item() == pytest.approx(1.0, abs=1e-12), (
                f"max|output(size={size})| = {x.abs().max().item()} "
                f"(expected exactly 1.0)"
            )
