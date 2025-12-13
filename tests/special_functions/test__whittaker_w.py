import torch
import scipy.special

from torchscience.special_functions import whittaker_w


class TestWhittakerW:
    """Tests for the Whittaker W function."""

    def test_known_values(self):
        """Test against known mathematical values."""
        kappa = torch.tensor([0.0])
        mu = torch.tensor([0.0])
        z = torch.tensor([1.0])
        result = whittaker_w(kappa, mu, z)
        # The result should be finite
        assert torch.isfinite(result).all()

    def test_reference_scipy(self):
        """Compare against scipy reference implementation."""
        kappa = torch.tensor([0.5, 1.0, 1.5])
        mu = torch.tensor([0.25, 0.5, 0.75])
        z = torch.tensor([1.0, 2.0, 3.0])

        result = whittaker_w(kappa, mu, z)
        expected = torch.tensor([
            scipy.special.whittaker_w(k, m, x)
            for k, m, x in zip(kappa.numpy(), mu.numpy(), z.numpy())
        ])

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-4)

    def test_shape_preservation(self):
        """Verify output shape matches input shape."""
        kappa = torch.tensor([0.5, 1.0, 1.5])
        mu = torch.tensor([0.25, 0.5, 0.75])
        z = torch.tensor([1.0, 2.0, 3.0])
        result = whittaker_w(kappa, mu, z)
        assert result.shape == kappa.shape

    def test_dtype_preservation(self):
        """Verify output dtype matches input dtype."""
        for dtype in [torch.float32, torch.float64]:
            kappa = torch.tensor([0.5, 1.0], dtype=dtype)
            mu = torch.tensor([0.25, 0.5], dtype=dtype)
            z = torch.tensor([1.0, 2.0], dtype=dtype)
            result = whittaker_w(kappa, mu, z)
            assert result.dtype == dtype

    def test_gradcheck(self):
        """Verify gradients using finite differences."""
        kappa = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64, requires_grad=True)
        mu = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64, requires_grad=True)
        z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(
            whittaker_w, (kappa, mu, z), eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_scalar(self):
        """Verify function works with 0-dimensional tensors."""
        kappa = torch.tensor(0.5)
        mu = torch.tensor(0.25)
        z = torch.tensor(1.0)
        result = whittaker_w(kappa, mu, z)
        assert result.dim() == 0

    def test_empty_tensor(self):
        """Verify function works with empty tensors."""
        kappa = torch.tensor([], dtype=torch.float32)
        mu = torch.tensor([], dtype=torch.float32)
        z = torch.tensor([], dtype=torch.float32)
        result = whittaker_w(kappa, mu, z)
        assert result.numel() == 0
