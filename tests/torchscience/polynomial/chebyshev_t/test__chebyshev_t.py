"""Tests for ChebyshevT tensorclass."""

import pytest
import torch

from torchscience.polynomial import chebyshev_t


class TestChebyshevTConstructor:
    """Tests for chebyshev_t() constructor."""

    def test_single_coefficient(self):
        """Constant Chebyshev series."""
        c = chebyshev_t(torch.tensor([3.0]))
        assert c.coeffs.shape == (1,)
        assert c.coeffs[0] == 3.0

    def test_multiple_coefficients(self):
        """Standard Chebyshev series."""
        c = chebyshev_t(torch.tensor([1.0, 2.0, 3.0]))
        assert c.coeffs.shape == (3,)
        torch.testing.assert_close(c.coeffs, torch.tensor([1.0, 2.0, 3.0]))

    def test_empty_raises(self):
        """Empty coefficients raise error."""
        from torchscience.polynomial import PolynomialError

        with pytest.raises(PolynomialError):
            chebyshev_t(torch.tensor([]))

    def test_preserves_dtype(self):
        """Dtype is preserved."""
        c = chebyshev_t(torch.tensor([1.0, 2.0], dtype=torch.float64))
        assert c.coeffs.dtype == torch.float64

    def test_preserves_device(self):
        """Device is preserved."""
        coeffs = torch.tensor([1.0, 2.0])
        c = chebyshev_t(coeffs)
        assert c.coeffs.device == coeffs.device
