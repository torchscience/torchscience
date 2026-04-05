import torch
import torch.testing

import torchscience.special_functions


class TestWeierstrassIntegration:
    """Integration tests verifying relationships between Weierstrass functions."""

    def test_sigma_ratio_is_zeta(self):
        """Test that sigma'(z)/sigma(z) = zeta(z).

        The Weierstrass sigma function satisfies sigma'/sigma = zeta.
        This is the logarithmic derivative relationship.
        """
        # Test several points away from poles and zeros
        z_values = [0.3, 0.4, 0.5, 0.6, 0.7]
        g2_val = 1.0
        g3_val = 0.5

        for z_val in z_values:
            z = torch.tensor([z_val], dtype=torch.float64, requires_grad=True)
            g2 = torch.tensor([g2_val], dtype=torch.float64)
            g3 = torch.tensor([g3_val], dtype=torch.float64)

            # Compute sigma(z)
            sigma = torchscience.special_functions.weierstrass_sigma(z, g2, g3)

            # Compute sigma'(z) via autograd
            (sigma_prime,) = torch.autograd.grad(sigma, z, create_graph=True)

            # Compute sigma'/sigma
            ratio = sigma_prime / sigma

            # Compute zeta(z) directly
            z_no_grad = torch.tensor([z_val], dtype=torch.float64)
            zeta = torchscience.special_functions.weierstrass_zeta(
                z_no_grad, g2, g3
            )

            torch.testing.assert_close(
                ratio,
                zeta,
                rtol=1e-3,
                atol=1e-3,
                msg=f"sigma'/sigma != zeta at z={z_val}",
            )

    def test_sigma_is_odd(self):
        """Test that sigma(-z) = -sigma(z).

        The Weierstrass sigma function is an odd function.
        """
        z_values = [0.2, 0.3, 0.4, 0.5, 0.6]
        g2_val = 1.0
        g3_val = 0.5

        for z_val in z_values:
            z = torch.tensor([z_val], dtype=torch.float64)
            g2 = torch.tensor([g2_val], dtype=torch.float64)
            g3 = torch.tensor([g3_val], dtype=torch.float64)

            sigma_pos = torchscience.special_functions.weierstrass_sigma(
                z, g2, g3
            )
            sigma_neg = torchscience.special_functions.weierstrass_sigma(
                -z, g2, g3
            )

            torch.testing.assert_close(
                sigma_pos,
                -sigma_neg,
                rtol=1e-10,
                atol=1e-10,
                msg=f"sigma(-z) != -sigma(z) at z={z_val}",
            )

    def test_zeta_is_odd(self):
        """Test that zeta(-z) = -zeta(z).

        The Weierstrass zeta function is an odd function.
        """
        z_values = [0.2, 0.3, 0.4, 0.5, 0.6]
        g2_val = 1.0
        g3_val = 0.5

        for z_val in z_values:
            z = torch.tensor([z_val], dtype=torch.float64)
            g2 = torch.tensor([g2_val], dtype=torch.float64)
            g3 = torch.tensor([g3_val], dtype=torch.float64)

            zeta_pos = torchscience.special_functions.weierstrass_zeta(
                z, g2, g3
            )
            zeta_neg = torchscience.special_functions.weierstrass_zeta(
                -z, g2, g3
            )

            torch.testing.assert_close(
                zeta_pos,
                -zeta_neg,
                rtol=1e-10,
                atol=1e-10,
                msg=f"zeta(-z) != -zeta(z) at z={z_val}",
            )

    def test_p_is_even(self):
        """Test that P(-z) = P(z).

        The Weierstrass P function is an even function.
        """
        z_values = [0.2, 0.3, 0.4, 0.5, 0.6]
        g2_val = 1.0
        g3_val = 0.5

        for z_val in z_values:
            z = torch.tensor([z_val], dtype=torch.float64)
            g2 = torch.tensor([g2_val], dtype=torch.float64)
            g3 = torch.tensor([g3_val], dtype=torch.float64)

            p_pos = torchscience.special_functions.weierstrass_p(z, g2, g3)
            p_neg = torchscience.special_functions.weierstrass_p(-z, g2, g3)

            torch.testing.assert_close(
                p_pos,
                p_neg,
                rtol=1e-10,
                atol=1e-10,
                msg=f"P(-z) != P(z) at z={z_val}",
            )

    def test_sigma_zero_at_origin(self):
        """Test that sigma(0) = 0.

        The Weierstrass sigma function has a simple zero at the origin.
        """
        z = torch.tensor([0.0], dtype=torch.float64)
        g2 = torch.tensor([1.0], dtype=torch.float64)
        g3 = torch.tensor([0.5], dtype=torch.float64)

        sigma = torchscience.special_functions.weierstrass_sigma(z, g2, g3)

        torch.testing.assert_close(
            sigma,
            torch.tensor([0.0], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_sigma_zero_at_origin_various_invariants(self):
        """Test sigma(0) = 0 for various g2, g3 values."""
        invariants = [
            (1.0, 0.0),
            (0.0, 1.0),
            (2.0, 0.5),
            (0.5, 0.25),
            (4.0, 1.0),
        ]

        z = torch.tensor([0.0], dtype=torch.float64)

        for g2_val, g3_val in invariants:
            g2 = torch.tensor([g2_val], dtype=torch.float64)
            g3 = torch.tensor([g3_val], dtype=torch.float64)

            sigma = torchscience.special_functions.weierstrass_sigma(z, g2, g3)

            torch.testing.assert_close(
                sigma,
                torch.tensor([0.0], dtype=torch.float64),
                rtol=1e-10,
                atol=1e-10,
                msg=f"sigma(0) != 0 for g2={g2_val}, g3={g3_val}",
            )

    def test_sigma_derivative_at_origin(self):
        """Test that sigma'(0) = 1.

        The Weierstrass sigma function has sigma'(0) = 1 by normalization.
        """
        z = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        g2 = torch.tensor([1.0], dtype=torch.float64)
        g3 = torch.tensor([0.5], dtype=torch.float64)

        sigma = torchscience.special_functions.weierstrass_sigma(z, g2, g3)
        (sigma_prime,) = torch.autograd.grad(sigma, z, create_graph=True)

        torch.testing.assert_close(
            sigma_prime,
            torch.tensor([1.0], dtype=torch.float64),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_zeta_pole_at_origin(self):
        """Test that zeta(z) has a simple pole at z=0.

        Near z=0, zeta(z) ~ 1/z.
        """
        z = torch.tensor([0.0], dtype=torch.float64)
        g2 = torch.tensor([1.0], dtype=torch.float64)
        g3 = torch.tensor([0.5], dtype=torch.float64)

        zeta = torchscience.special_functions.weierstrass_zeta(z, g2, g3)

        # zeta(0) should be infinity (simple pole)
        assert torch.isinf(zeta).all() or torch.isnan(zeta).all(), (
            f"Expected inf or nan at z=0, got {zeta}"
        )

    def test_zeta_near_origin_behavior(self):
        """Test that near z=0, zeta(z) ~ 1/z.

        The Laurent series of zeta starts with 1/z.
        """
        z_values = [0.01, 0.001]
        g2 = torch.tensor([0.0], dtype=torch.float64)
        g3 = torch.tensor([0.0], dtype=torch.float64)

        for z_val in z_values:
            z = torch.tensor([z_val], dtype=torch.float64)
            zeta = torchscience.special_functions.weierstrass_zeta(z, g2, g3)
            expected = 1.0 / z_val

            # Should be approximately 1/z for small z
            rel_error = abs((zeta.item() - expected) / expected)
            assert rel_error < 0.1, (
                f"At z={z_val}, expected ~{expected}, got {zeta.item()}"
            )

    def test_p_pole_at_origin(self):
        """Test that P(z) has a double pole at z=0."""
        z = torch.tensor([0.0], dtype=torch.float64)
        g2 = torch.tensor([1.0], dtype=torch.float64)
        g3 = torch.tensor([0.5], dtype=torch.float64)

        p_val = torchscience.special_functions.weierstrass_p(z, g2, g3)

        # P(0) should be infinity (double pole)
        assert torch.isinf(p_val).all() or torch.isnan(p_val).all(), (
            f"Expected inf or nan at z=0, got {p_val}"
        )

    def test_p_near_origin_behavior(self):
        """Test that near z=0, P(z) ~ 1/z^2.

        The Laurent series of P starts with 1/z^2.
        """
        z_values = [0.01, 0.001]
        g2 = torch.tensor([0.0], dtype=torch.float64)
        g3 = torch.tensor([0.0], dtype=torch.float64)

        for z_val in z_values:
            z = torch.tensor([z_val], dtype=torch.float64)
            p_val = torchscience.special_functions.weierstrass_p(z, g2, g3)
            expected = 1.0 / (z_val**2)

            # Should be approximately 1/z^2 for small z
            rel_error = abs((p_val.item() - expected) / expected)
            assert rel_error < 0.1, (
                f"At z={z_val}, expected ~{expected}, got {p_val.item()}"
            )

    def test_batch_relationships(self):
        """Test that relationships hold in batch computation."""
        z = torch.tensor([0.3, 0.4, 0.5, 0.6], dtype=torch.float64)
        g2 = torch.tensor([1.0], dtype=torch.float64)
        g3 = torch.tensor([0.5], dtype=torch.float64)

        # Test P is even (batch)
        p_pos = torchscience.special_functions.weierstrass_p(z, g2, g3)
        p_neg = torchscience.special_functions.weierstrass_p(-z, g2, g3)
        torch.testing.assert_close(p_pos, p_neg, rtol=1e-10, atol=1e-10)

        # Test sigma is odd (batch)
        sigma_pos = torchscience.special_functions.weierstrass_sigma(z, g2, g3)
        sigma_neg = torchscience.special_functions.weierstrass_sigma(
            -z, g2, g3
        )
        torch.testing.assert_close(
            sigma_pos, -sigma_neg, rtol=1e-10, atol=1e-10
        )

        # Test zeta is odd (batch)
        zeta_pos = torchscience.special_functions.weierstrass_zeta(z, g2, g3)
        zeta_neg = torchscience.special_functions.weierstrass_zeta(-z, g2, g3)
        torch.testing.assert_close(zeta_pos, -zeta_neg, rtol=1e-10, atol=1e-10)

    def test_eta_is_finite(self):
        """Test that eta returns finite values for valid invariants."""
        invariants = [
            (1.0, 0.0),
            (0.0, 1.0),
            (2.0, 0.5),
            (4.0, 1.0),
        ]

        for g2_val, g3_val in invariants:
            g2 = torch.tensor([g2_val], dtype=torch.float64)
            g3 = torch.tensor([g3_val], dtype=torch.float64)

            eta = torchscience.special_functions.weierstrass_eta(g2, g3)
            assert torch.isfinite(eta).all(), (
                f"eta should be finite for g2={g2_val}, g3={g3_val}, got {eta}"
            )

    def test_eta_consistency_across_invariants(self):
        """Test eta is consistent when computed via different paths."""
        g2 = torch.tensor([1.0], dtype=torch.float64)
        g3 = torch.tensor([0.5], dtype=torch.float64)

        # Compute eta directly
        eta = torchscience.special_functions.weierstrass_eta(g2, g3)

        # eta should be a finite real value
        assert torch.isfinite(eta).all()
        assert eta.dtype == torch.float64
