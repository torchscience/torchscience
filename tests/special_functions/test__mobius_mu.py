import torch

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import mobius_mu


class TestMobiusMu(UnaryOperatorTestCase):
    func = staticmethod(mobius_mu)
    op_name = "_mobius_mu"

    symmetry = None
    period = None

    # Known values for Mobius function
    # mu(1) = 1
    # mu(2) = -1 (prime)
    # mu(3) = -1 (prime)
    # mu(4) = 0 (4 = 2^2, has squared factor)
    # mu(5) = -1 (prime)
    # mu(6) = 1 (6 = 2*3, two distinct primes)
    # mu(8) = 0 (8 = 2^3)
    # mu(10) = 1 (10 = 2*5)
    known_values = {
        1.0: 1.0,
        2.0: -1.0,
        3.0: -1.0,
        4.0: 0.0,
        5.0: -1.0,
        6.0: 1.0,
        7.0: -1.0,
        8.0: 0.0,
        9.0: 0.0,
        10.0: 1.0,
        30.0: -1.0,  # 30 = 2*3*5, three distinct primes
    }

    reference = None

    input_range = (1.0, 30.0)
    gradcheck_inputs = [1.0, 2.0, 3.0, 5.0]

    supports_complex = False

    def test_mu_of_primes(self):
        """Test mu(p) = -1 for primes p."""
        primes = torch.tensor([2.0, 3.0, 5.0, 7.0, 11.0, 13.0])
        output = mobius_mu(primes)
        expected = -torch.ones_like(primes)
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_mu_of_prime_squares(self):
        """Test mu(p^2) = 0 for squared primes."""
        # 4 = 2^2, 9 = 3^2, 25 = 5^2, 49 = 7^2
        n = torch.tensor([4.0, 9.0, 25.0, 49.0])
        output = mobius_mu(n)
        expected = torch.zeros_like(n)
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_mu_of_products_of_two_primes(self):
        """Test mu(p*q) = 1 for distinct primes p, q."""
        # 6 = 2*3, 10 = 2*5, 15 = 3*5, 21 = 3*7
        n = torch.tensor([6.0, 10.0, 15.0, 21.0])
        output = mobius_mu(n)
        expected = torch.ones_like(n)
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_mu_of_products_of_three_primes(self):
        """Test mu(p*q*r) = -1 for distinct primes p, q, r."""
        # 30 = 2*3*5, 42 = 2*3*7
        n = torch.tensor([30.0, 42.0])
        output = mobius_mu(n)
        expected = -torch.ones_like(n)
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_mu_1(self):
        """Test mu(1) = 1."""
        n = torch.tensor([1.0])
        output = mobius_mu(n)
        expected = torch.tensor([1.0])
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)
