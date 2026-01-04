import scipy.special
import torch

import torchscience.special_functions  # noqa: F401 - registers operators


class TestRegularizedGammaPForward:
    """Test regularized_gamma_p forward correctness."""

    def test_basic_values(self):
        """Test against scipy.special.gammainc."""
        a = torch.tensor([1.0, 2.0, 3.0, 5.0])
        x = torch.tensor([0.5, 1.0, 2.0, 3.0])

        result = torch.ops.torchscience.regularized_gamma_p(a, x)

        expected = torch.tensor(
            scipy.special.gammainc(a.numpy(), x.numpy()),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-6)
