import mpmath
import pytest
import torch
import torch.testing

import torchscience.special_functions


class TestConfluentHypergeometricUIntegerB:
    """Tests for U(a, b, z) when b is a positive integer."""

    @pytest.mark.parametrize(
        "a_val, b_val, z_val",
        [
            # b=2 cases (simplest integer b >= 2)
            (0.5, 2.0, 1.0),
            (1.5, 2.0, 2.0),
            (0.5, 2.0, 0.5),
            # b=3 cases
            (0.5, 3.0, 1.0),
            (1.0, 3.0, 2.0),
            (2.0, 3.0, 0.5),
            # b=4 case
            (1.0, 4.0, 1.5),
            # b=1 case (boundary)
            (0.5, 1.0, 1.0),
            (1.5, 1.0, 2.0),
        ],
    )
    def test_mpmath_reference(self, a_val, b_val, z_val):
        """Test U(a, b, z) against mpmath for integer b values."""
        expected = float(mpmath.hyperu(a_val, b_val, z_val))

        a = torch.tensor([a_val], dtype=torch.float64)
        b = torch.tensor([b_val], dtype=torch.float64)
        z = torch.tensor([z_val], dtype=torch.float64)

        result = torchscience.special_functions.confluent_hypergeometric_u(
            a, b, z
        )

        torch.testing.assert_close(
            result,
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-6,
            atol=1e-10,
        )
