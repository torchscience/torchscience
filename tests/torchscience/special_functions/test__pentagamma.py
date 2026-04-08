import math

import sympy
import torch
import torch.testing
from sympy import I, N

import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
    ToleranceConfig,
)


def sympy_pentagamma(z: float | complex) -> float | complex:
    if isinstance(z, complex):
        sympy_z = sympy.Float(z.real) + I * sympy.Float(z.imag)
    else:
        sympy_z = sympy.Float(z)
    result = N(sympy.polygamma(3, sympy_z), 50)
    if result.is_real:
        return float(result)
    return complex(result)


class TestPentagamma(OpTestCase):
    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="pentagamma",
            func=torchscience.special_functions.pentagamma,
            arity=1,
            input_specs=[
                InputSpec(
                    name="z",
                    position=0,
                    default_real_range=(0.5, 20.0),
                    excluded_values={0.0, -1.0, -2.0, -3.0, -4.0, -5.0},
                ),
            ],
            sympy_func=lambda z: sympy.polygamma(3, z),
            tolerances=ToleranceConfig(
                float64_rtol=1e-8,
                float64_atol=1e-8,
                sympy_rtol=1e-8,
                sympy_atol=1e-8,
            ),
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_gradcheck_complex",
                "test_gradgradcheck_complex",
                "test_sparse_coo_basic",
                "test_low_precision_forward",
                "test_sympy_reference_complex",
            },
            special_values=[
                SpecialValue(
                    inputs=(1.0,),
                    expected=math.pi**4 / 15,
                    rtol=1e-6,
                    atol=1e-6,
                    description="psi_3(1) = pi^4/15",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )

    def test_gradcheck(self):
        z = torch.tensor(
            [1.5, 2.5, 5.0], dtype=torch.float64, requires_grad=True
        )
        torch.autograd.gradcheck(
            torchscience.special_functions.pentagamma,
            (z,),
            eps=1e-5,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_gradgradcheck(self):
        z = torch.tensor([5.0, 10.0], dtype=torch.float64, requires_grad=True)
        torch.autograd.gradgradcheck(
            torchscience.special_functions.pentagamma,
            (z,),
            eps=1e-4,
            atol=1e-1,
            rtol=1e-1,
        )
