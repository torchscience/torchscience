"""Cross-operator relationship tests for information theory module.

Tests the mathematical relationships between entropy and divergence operators:
- H(P, Q) = H(P) + D_KL(P || Q)  (cross-entropy decomposition)
- D_KL(P || Q) <= log(1 + chi^2(P || Q))  (Pinsker-type bound)
- D_JS(P, Q) <= (D_KL(P || M) + D_KL(Q || M)) / 2  where M = (P + Q) / 2
- H(P, P) = H(P)  (self cross-entropy)
- D_JS(P, Q) = D_JS(Q, P)  (JS symmetry)
"""

import torch

from torchscience.information_theory import (
    chi_squared_divergence,
    cross_entropy,
    jensen_shannon_divergence,
    kullback_leibler_divergence,
    shannon_entropy,
)


class TestCrossEntropyDecomposition:
    """Test H(P, Q) = H(P) + D_KL(P || Q)."""

    def test_decomposition_1d(self):
        """Decomposition holds for 1D distributions."""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)

        ce = cross_entropy(p, q)
        h = shannon_entropy(p)
        kl = kullback_leibler_divergence(p, q)

        assert torch.isclose(ce, h + kl, rtol=1e-5)

    def test_decomposition_batch(self):
        """Decomposition holds for batched distributions."""
        torch.manual_seed(42)
        p = torch.softmax(torch.randn(5, 8), dim=-1)
        q = torch.softmax(torch.randn(5, 8), dim=-1)

        ce = cross_entropy(p, q)
        h = shannon_entropy(p)
        kl = kullback_leibler_divergence(p, q)

        assert torch.allclose(ce, h + kl, rtol=1e-5)

    def test_decomposition_random(self):
        """Decomposition holds for random distributions."""
        torch.manual_seed(123)
        for _ in range(10):
            p = torch.softmax(torch.randn(20), dim=-1)
            q = torch.softmax(torch.randn(20), dim=-1)

            ce = cross_entropy(p, q)
            h = shannon_entropy(p)
            kl = kullback_leibler_divergence(p, q)

            assert torch.isclose(ce, h + kl, rtol=1e-4), (
                f"Decomposition failed: CE={ce}, H={h}, KL={kl}"
            )


class TestSelfCrossEntropy:
    """Test H(P, P) = H(P)."""

    def test_self_cross_entropy_equals_entropy(self):
        """Self cross-entropy equals Shannon entropy."""
        p = torch.softmax(torch.randn(10), dim=-1)
        ce_pp = cross_entropy(p, p)
        h_p = shannon_entropy(p)
        assert torch.isclose(ce_pp, h_p, rtol=1e-5)

    def test_self_cross_entropy_batch(self):
        """Self cross-entropy equals Shannon entropy for batched inputs."""
        p = torch.softmax(torch.randn(5, 8), dim=-1)
        ce_pp = cross_entropy(p, p)
        h_p = shannon_entropy(p)
        assert torch.allclose(ce_pp, h_p, rtol=1e-5)


class TestKLChiSquaredBound:
    """Test D_KL(P || Q) <= log(1 + chi^2(P || Q))."""

    def test_kl_bounded_by_chi_squared(self):
        """KL divergence is bounded by chi-squared divergence."""
        torch.manual_seed(42)
        for _ in range(10):
            p = torch.softmax(torch.randn(10), dim=-1)
            q = torch.softmax(torch.randn(10), dim=-1)

            kl = kullback_leibler_divergence(p, q)
            chi2 = chi_squared_divergence(p, q)

            bound = torch.log(1 + chi2)
            assert kl <= bound + 1e-5, f"KL={kl} > bound={bound}"

    def test_kl_bounded_batch(self):
        """KL bound holds for batched inputs."""
        p = torch.softmax(torch.randn(5, 8), dim=-1)
        q = torch.softmax(torch.randn(5, 8), dim=-1)

        kl = kullback_leibler_divergence(p, q)
        chi2 = chi_squared_divergence(p, q)

        bounds = torch.log(1 + chi2)
        assert (kl <= bounds + 1e-5).all()


class TestJensenShannonSymmetry:
    """Test D_JS(P, Q) = D_JS(Q, P)."""

    def test_js_symmetric(self):
        """Jensen-Shannon divergence is symmetric."""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)

        js_pq = jensen_shannon_divergence(p, q)
        js_qp = jensen_shannon_divergence(q, p)

        assert torch.isclose(js_pq, js_qp, rtol=1e-5)

    def test_js_symmetric_batch(self):
        """Jensen-Shannon divergence is symmetric for batched inputs."""
        p = torch.softmax(torch.randn(5, 8), dim=-1)
        q = torch.softmax(torch.randn(5, 8), dim=-1)

        js_pq = jensen_shannon_divergence(p, q)
        js_qp = jensen_shannon_divergence(q, p)

        assert torch.allclose(js_pq, js_qp, rtol=1e-5)


class TestJensenShannonBounds:
    """Test bounds on Jensen-Shannon divergence."""

    def test_js_bounded_by_log2(self):
        """JS divergence in nats is bounded by ln(2)."""
        torch.manual_seed(42)
        for _ in range(10):
            p = torch.softmax(torch.randn(10), dim=-1)
            q = torch.softmax(torch.randn(10), dim=-1)

            js = jensen_shannon_divergence(p, q)
            assert js <= torch.log(torch.tensor(2.0)) + 1e-5

    def test_js_non_negative(self):
        """JS divergence is non-negative."""
        torch.manual_seed(42)
        for _ in range(10):
            p = torch.softmax(torch.randn(10), dim=-1)
            q = torch.softmax(torch.randn(10), dim=-1)

            js = jensen_shannon_divergence(p, q)
            assert js >= -1e-6


class TestIdenticalDistributions:
    """Test all divergences are zero for identical distributions."""

    def test_kl_zero_for_identical(self):
        """KL divergence is zero when P = Q."""
        p = torch.softmax(torch.randn(10), dim=-1)
        kl = kullback_leibler_divergence(p, p)
        assert torch.isclose(kl, torch.tensor(0.0), atol=1e-6)

    def test_js_zero_for_identical(self):
        """JS divergence is zero when P = Q."""
        p = torch.softmax(torch.randn(10), dim=-1)
        js = jensen_shannon_divergence(p, p)
        assert torch.isclose(js, torch.tensor(0.0), atol=1e-6)

    def test_chi_squared_zero_for_identical(self):
        """Chi-squared divergence is zero when P = Q."""
        p = torch.softmax(torch.randn(10), dim=-1)
        chi2 = chi_squared_divergence(p, p)
        assert torch.isclose(chi2, torch.tensor(0.0), atol=1e-6)


class TestNonNegativity:
    """Test non-negativity of all measures."""

    def test_all_measures_non_negative(self):
        """All entropy and divergence measures are non-negative."""
        torch.manual_seed(42)
        for _ in range(10):
            p = torch.softmax(torch.randn(10), dim=-1)
            q = torch.softmax(torch.randn(10), dim=-1)

            h = shannon_entropy(p)
            ce = cross_entropy(p, q)
            kl = kullback_leibler_divergence(p, q)
            js = jensen_shannon_divergence(p, q)
            chi2 = chi_squared_divergence(p, q)

            assert h >= -1e-6, f"Shannon entropy negative: {h}"
            assert ce >= -1e-6, f"Cross-entropy negative: {ce}"
            assert kl >= -1e-6, f"KL divergence negative: {kl}"
            assert js >= -1e-6, f"JS divergence negative: {js}"
            assert chi2 >= -1e-6, f"Chi-squared divergence negative: {chi2}"


class TestUniformDistributions:
    """Test special cases with uniform distributions."""

    def test_uniform_max_entropy(self):
        """Uniform distribution has maximum entropy log(n)."""
        for n in [2, 4, 8, 16]:
            p = torch.ones(n) / n
            h = shannon_entropy(p)
            expected = torch.log(torch.tensor(float(n)))
            assert torch.isclose(h, expected, rtol=1e-5)

    def test_uniform_zero_divergences(self):
        """All divergences are zero between identical uniforms."""
        p = torch.ones(8) / 8
        q = torch.ones(8) / 8

        assert torch.isclose(
            kullback_leibler_divergence(p, q), torch.tensor(0.0), atol=1e-6
        )
        assert torch.isclose(
            jensen_shannon_divergence(p, q), torch.tensor(0.0), atol=1e-6
        )
        assert torch.isclose(
            chi_squared_divergence(p, q), torch.tensor(0.0), atol=1e-6
        )
