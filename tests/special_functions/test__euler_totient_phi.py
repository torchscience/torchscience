import torch

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import euler_totient_phi


class TestEulerTotientPhi(UnaryOperatorTestCase):
    func = staticmethod(euler_totient_phi)
    op_name = "_euler_totient_phi"

    symmetry = None
    period = None

    # Known values for Euler's totient function
    known_values = {
        1.0: 1.0,
        2.0: 1.0,
        3.0: 2.0,
        4.0: 2.0,
        5.0: 4.0,
        6.0: 2.0,
        7.0: 6.0,
        8.0: 4.0,
        9.0: 6.0,
        10.0: 4.0,
        12.0: 4.0,
    }

    reference = None

    input_range = (1.0, 20.0)
    gradcheck_inputs = [1.0, 2.0, 3.0, 5.0]

    supports_complex = False

    def test_phi_of_primes(self):
        """Test phi(p) = p - 1 for primes p."""
        primes = torch.tensor([2.0, 3.0, 5.0, 7.0, 11.0, 13.0])
        output = euler_totient_phi(primes)
        expected = primes - 1
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_phi_of_prime_powers(self):
        """Test phi(p^k) = p^(k-1) * (p-1) for prime powers."""
        # phi(4) = phi(2^2) = 2^1 * (2-1) = 2
        # phi(8) = phi(2^3) = 2^2 * (2-1) = 4
        # phi(9) = phi(3^2) = 3^1 * (3-1) = 6
        # phi(27) = phi(3^3) = 3^2 * (3-1) = 18
        n = torch.tensor([4.0, 8.0, 9.0, 27.0])
        output = euler_totient_phi(n)
        expected = torch.tensor([2.0, 4.0, 6.0, 18.0])
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_multiplicativity(self):
        """Test phi(mn) = phi(m) * phi(n) when gcd(m,n) = 1."""
        # phi(6) = phi(2) * phi(3) = 1 * 2 = 2
        # phi(15) = phi(3) * phi(5) = 2 * 4 = 8
        # phi(35) = phi(5) * phi(7) = 4 * 6 = 24
        n = torch.tensor([6.0, 15.0, 35.0])
        output = euler_totient_phi(n)
        expected = torch.tensor([2.0, 8.0, 24.0])
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_phi_1(self):
        """Test phi(1) = 1."""
        n = torch.tensor([1.0])
        output = euler_totient_phi(n)
        expected = torch.tensor([1.0])
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)
