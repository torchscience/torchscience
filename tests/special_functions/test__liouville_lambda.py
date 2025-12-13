import torch

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import liouville_lambda


class TestLiouvilleLambda(UnaryOperatorTestCase):
    func = staticmethod(liouville_lambda)
    op_name = "_liouville_lambda"

    symmetry = None
    period = None

    # Known values for Liouville function
    # lambda(n) = (-1)^Omega(n) where Omega(n) counts prime factors with multiplicity
    # lambda(1) = 1 (Omega(1) = 0)
    # lambda(2) = -1 (Omega(2) = 1, prime)
    # lambda(3) = -1 (Omega(3) = 1, prime)
    # lambda(4) = 1 (Omega(4) = 2, since 4 = 2^2)
    # lambda(5) = -1 (Omega(5) = 1, prime)
    # lambda(6) = 1 (Omega(6) = 2, since 6 = 2*3)
    # lambda(8) = -1 (Omega(8) = 3, since 8 = 2^3)
    # lambda(12) = -1 (Omega(12) = 3, since 12 = 2^2 * 3)
    known_values = {
        1.0: 1.0,
        2.0: -1.0,
        3.0: -1.0,
        4.0: 1.0,
        5.0: -1.0,
        6.0: 1.0,
        7.0: -1.0,
        8.0: -1.0,
        9.0: 1.0,
        10.0: 1.0,
        12.0: -1.0,
        16.0: 1.0,
        30.0: -1.0,  # 30 = 2*3*5, Omega(30) = 3
    }

    reference = None

    input_range = (1.0, 30.0)
    gradcheck_inputs = [1.0, 2.0, 3.0, 4.0]

    supports_complex = False

    def test_lambda_of_primes(self):
        """Test lambda(p) = -1 for primes p."""
        primes = torch.tensor([2.0, 3.0, 5.0, 7.0, 11.0, 13.0])
        output = liouville_lambda(primes)
        expected = -torch.ones_like(primes)
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_lambda_of_prime_squares(self):
        """Test lambda(p^2) = 1 for squared primes."""
        # 4 = 2^2, 9 = 3^2, 25 = 5^2, 49 = 7^2
        n = torch.tensor([4.0, 9.0, 25.0, 49.0])
        output = liouville_lambda(n)
        expected = torch.ones_like(n)
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_lambda_of_prime_cubes(self):
        """Test lambda(p^3) = -1 for cubed primes."""
        # 8 = 2^3, 27 = 3^3
        n = torch.tensor([8.0, 27.0])
        output = liouville_lambda(n)
        expected = -torch.ones_like(n)
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_lambda_multiplicativity(self):
        """Test that lambda is completely multiplicative: lambda(mn) = lambda(m)*lambda(n)."""
        # Test with coprime pairs
        m_vals = torch.tensor([2.0, 3.0, 4.0, 5.0])
        n_vals = torch.tensor([3.0, 4.0, 5.0, 6.0])
        mn_vals = m_vals * n_vals

        lambda_m = liouville_lambda(m_vals)
        lambda_n = liouville_lambda(n_vals)
        lambda_mn = liouville_lambda(mn_vals)

        torch.testing.assert_close(lambda_mn, lambda_m * lambda_n, atol=1e-6, rtol=1e-5)

    def test_lambda_1(self):
        """Test lambda(1) = 1."""
        n = torch.tensor([1.0])
        output = liouville_lambda(n)
        expected = torch.tensor([1.0])
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)
