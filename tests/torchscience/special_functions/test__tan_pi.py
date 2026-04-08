import sympy
import torch
import torch.testing

import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
    ToleranceConfig,
)


class TestTanPi(OpTestCase):
    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="tan_pi",
            func=torchscience.special_functions.tan_pi,
            arity=1,
            input_specs=[
                InputSpec(
                    name="x",
                    position=0,
                    default_real_range=(-0.49, 0.49),
                ),
            ],
            sympy_func=lambda x: sympy.tan(sympy.pi * x),
            tolerances=ToleranceConfig(
                float64_rtol=1e-10,
                float64_atol=1e-10,
                sympy_rtol=1e-10,
                sympy_atol=1e-10,
            ),
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_sparse_coo_basic",
                "test_low_precision_forward",
                "test_gradcheck_complex",
                "test_gradgradcheck_complex",
                "test_sympy_reference_complex",
            },
            special_values=[
                SpecialValue(
                    inputs=(0.0,),
                    expected=0.0,
                    rtol=1e-10,
                    atol=1e-10,
                    description="tan(0) = 0",
                ),
                SpecialValue(
                    inputs=(0.25,),
                    expected=1.0,
                    rtol=1e-10,
                    atol=1e-10,
                    description="tan(pi/4) = 1",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )

    def test_gradcheck(self):
        x = torch.tensor(
            [0.1, 0.2, 0.3], dtype=torch.float64, requires_grad=True
        )
        torch.autograd.gradcheck(
            torchscience.special_functions.tan_pi,
            (x,),
            eps=1e-5,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_gradgradcheck(self):
        x = torch.tensor([0.1, 0.3], dtype=torch.float64, requires_grad=True)
        torch.autograd.gradgradcheck(
            torchscience.special_functions.tan_pi,
            (x,),
            eps=1e-5,
            atol=1e-4,
            rtol=1e-4,
        )
