import math

import torchscience.special_functions
from torchscience.testing import (
    IdentitySpec,
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
    ToleranceConfig,
)


def _homogeneity(func):
    """Check R_F(ax, ay, az) = a^(-1/2) R_F(x, y, z)."""
    import torch

    x = torch.tensor([1.0, 2.0], dtype=torch.float64)
    y = torch.tensor([2.0, 3.0], dtype=torch.float64)
    z = torch.tensor([3.0, 4.0], dtype=torch.float64)
    a = 4.0
    left = func(a * x, a * y, a * z)
    right = func(x, y, z) / math.sqrt(a)
    return left, right


def _symmetry_xy(func):
    """Check R_F(x, y, z) = R_F(y, x, z)."""
    import torch

    x = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
    y = torch.tensor([2.0, 3.0, 1.5], dtype=torch.float64)
    z = torch.tensor([3.0, 4.0, 2.5], dtype=torch.float64)
    left = func(x, y, z)
    right = func(y, x, z)
    return left, right


class TestCarlsonEllipticIntegralRF(OpTestCase):
    """Tests for the carlson_elliptic_integral_r_f function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="carlson_elliptic_integral_r_f",
            func=torchscience.special_functions.carlson_elliptic_integral_r_f,
            arity=3,
            input_specs=[
                InputSpec(name="x", position=0, default_real_range=(0.1, 5.0)),
                InputSpec(name="y", position=1, default_real_range=(0.1, 5.0)),
                InputSpec(name="z", position=2, default_real_range=(0.1, 5.0)),
            ],
            tolerances=ToleranceConfig(),
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_gradcheck_complex",
                "test_gradgradcheck_real",
                "test_gradgradcheck_complex",
            },
            special_values=[
                SpecialValue(
                    inputs=(0.0, 1.0, 1.0),
                    expected=math.pi / 2,
                    description="R_F(0, 1, 1) = pi/2",
                ),
                SpecialValue(
                    inputs=(1.0, 1.0, 1.0),
                    expected=1.0,
                    description="R_F(x, x, x) = 1/sqrt(x) for x=1",
                ),
                SpecialValue(
                    inputs=(4.0, 4.0, 4.0),
                    expected=0.5,
                    description="R_F(4, 4, 4) = 1/sqrt(4) = 0.5",
                ),
            ],
            functional_identities=[
                IdentitySpec(
                    name="homogeneity",
                    identity_fn=_homogeneity,
                    description="R_F(ax, ay, az) = a^(-1/2) R_F(x, y, z)",
                ),
                IdentitySpec(
                    name="symmetry_xy",
                    identity_fn=_symmetry_xy,
                    description="R_F(x, y, z) = R_F(y, x, z)",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )
