import torch

from torchscience.integration.initial_value_problem import dormand_prince_5


class TestDormandPrince5Basic:
    def test_exponential_decay(self):
        """Test against analytical solution: dy/dt = -y, y(0) = 1 => y(t) = exp(-t)"""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = dormand_prince_5(decay, y0, t_span=(0.0, 5.0))

        expected = torch.exp(torch.tensor([-5.0]))
        assert torch.allclose(y_final, expected, rtol=1e-4)

    def test_harmonic_oscillator(self):
        """Test 2D system: simple harmonic oscillator"""

        def oscillator(t, y):
            x, v = y[..., 0], y[..., 1]
            return torch.stack([v, -x], dim=-1)  # dx/dt = v, dv/dt = -x

        y0 = torch.tensor([1.0, 0.0])  # x=1, v=0
        y_final, interp = dormand_prince_5(
            oscillator, y0, t_span=(0.0, 2 * torch.pi)
        )

        # After one period, should return to initial state
        assert torch.allclose(y_final, y0, atol=1e-3)

    def test_returns_interpolant(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = dormand_prince_5(f, y0, t_span=(0.0, 1.0))

        # Interpolant should be callable
        y_mid = interp(0.5)
        assert y_mid.shape == y0.shape

    def test_interpolant_endpoints(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = dormand_prince_5(f, y0, t_span=(0.0, 1.0))

        # At t=0, should match y0
        assert torch.allclose(interp(0.0), y0, atol=1e-6)
        # At t=1, should match y_final
        assert torch.allclose(interp(1.0), y_final, atol=1e-6)

    def test_multiple_time_queries(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([1.0])
        y_final, interp = dormand_prince_5(f, y0, t_span=(0.0, 1.0))

        t_query = torch.linspace(0, 1, 10)
        trajectory = interp(t_query)
        assert trajectory.shape == (10, 1)

        # Should be monotonically decreasing
        for i in range(9):
            assert trajectory[i, 0] > trajectory[i + 1, 0]


class TestDormandPrince5Complex:
    def test_complex_exponential_decay(self):
        """Test complex ODE: dy/dt = -i*y, y(0) = 1 => y(t) = exp(-i*t)"""

        def f(t, y):
            return -1j * y

        y0 = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        y_final, interp = dormand_prince_5(f, y0, t_span=(0.0, torch.pi))

        expected = torch.exp(-1j * torch.tensor(torch.pi, dtype=torch.float64))
        assert torch.allclose(y_final, expected.unsqueeze(0), atol=1e-4)

    def test_schrodinger_like(self):
        """Test Schrodinger-like equation: dy/dt = -i*H*y"""
        H = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)

        def f(t, y):
            return -1j * H @ y

        psi0 = torch.tensor([1.0 + 0j, 0.0 + 0j], dtype=torch.complex128)
        y_final, _ = dormand_prince_5(f, psi0, t_span=(0.0, 1.0))

        # Check normalization preserved
        norm = torch.abs(y_final).pow(2).sum()
        assert torch.allclose(
            norm, torch.tensor(1.0, dtype=torch.float64), atol=1e-5
        )

    def test_complex_gradcheck(self):
        """Verify gradients work for complex states"""
        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def f(t, y):
            return -theta.to(y.dtype) * 1j * y

        y0 = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        y_final, _ = dormand_prince_5(f, y0, t_span=(0.0, 1.0))

        # Can compute gradients through complex operations
        loss = y_final.abs().sum()
        loss.backward()

        assert theta.grad is not None
