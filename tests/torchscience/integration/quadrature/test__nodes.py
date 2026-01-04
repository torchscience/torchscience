import pytest
import torch
from numpy.polynomial.legendre import leggauss


class TestGaussLegendreNodesWeights:
    @pytest.mark.parametrize("n", [2, 5, 10, 32, 64])
    def test_matches_numpy(self, n):
        """Compare with numpy's Gauss-Legendre implementation"""
        from torchscience.integration.quadrature._nodes import (
            gauss_legendre_nodes_weights,
        )

        nodes, weights = gauss_legendre_nodes_weights(n, dtype=torch.float64)
        np_nodes, np_weights = leggauss(n)

        assert torch.allclose(nodes, torch.tensor(np_nodes), rtol=1e-12)
        assert torch.allclose(weights, torch.tensor(np_weights), rtol=1e-12)

    def test_nodes_in_interval(self):
        """Nodes should be in [-1, 1]"""
        from torchscience.integration.quadrature._nodes import (
            gauss_legendre_nodes_weights,
        )

        nodes, _ = gauss_legendre_nodes_weights(100)

        assert (nodes >= -1).all()
        assert (nodes <= 1).all()

    def test_weights_sum_to_two(self):
        """Weights should sum to 2 (length of [-1, 1])"""
        from torchscience.integration.quadrature._nodes import (
            gauss_legendre_nodes_weights,
        )

        _, weights = gauss_legendre_nodes_weights(50, dtype=torch.float64)

        assert torch.allclose(
            weights.sum(), torch.tensor(2.0, dtype=torch.float64), rtol=1e-10
        )

    def test_exact_for_polynomial(self):
        """Should exactly integrate polynomials up to degree 2n-1"""
        from torchscience.integration.quadrature._nodes import (
            gauss_legendre_nodes_weights,
        )

        n = 5
        nodes, weights = gauss_legendre_nodes_weights(n, dtype=torch.float64)

        # Integrate x^8 from -1 to 1 (degree 8 < 2*5-1=9, so should be exact)
        # integral of x^8 from -1 to 1 = 2/9
        integrand = nodes**8
        result = (integrand * weights).sum()
        expected = torch.tensor(2 / 9, dtype=torch.float64)

        assert torch.allclose(result, expected, rtol=1e-10)

    def test_n_equals_1(self):
        """Single-point quadrature: midpoint rule"""
        from torchscience.integration.quadrature._nodes import (
            gauss_legendre_nodes_weights,
        )

        nodes, weights = gauss_legendre_nodes_weights(1, dtype=torch.float64)

        assert nodes.shape == (1,)
        assert weights.shape == (1,)
        assert torch.allclose(nodes, torch.tensor([0.0], dtype=torch.float64))
        assert torch.allclose(
            weights, torch.tensor([2.0], dtype=torch.float64)
        )

    def test_invalid_n_raises(self):
        """n < 1 should raise ValueError"""
        from torchscience.integration.quadrature._nodes import (
            gauss_legendre_nodes_weights,
        )

        with pytest.raises(ValueError, match="at least 1"):
            gauss_legendre_nodes_weights(0)

    def test_device_placement(self):
        """Test that device parameter works"""
        from torchscience.integration.quadrature._nodes import (
            gauss_legendre_nodes_weights,
        )

        nodes, weights = gauss_legendre_nodes_weights(
            10, device=torch.device("cpu")
        )

        assert nodes.device == torch.device("cpu")
        assert weights.device == torch.device("cpu")
