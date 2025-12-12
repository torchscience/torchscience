import torch

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import sin_pi, cos_pi


class TestSinPi(UnaryOperatorTestCase):
    func = staticmethod(sin_pi)
    op_name = "_sin_pi"

    symmetry = "odd"
    period = 2.0
    bounds = (-1.0, 1.0)

    known_values = {
        0.0: 0.0,
        0.5: 1.0,
        1.0: 0.0,
        1.5: -1.0,
        2.0: 0.0,
    }
    zeros = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

    reference = staticmethod(lambda x: torch.sin(torch.pi * x))

    reference_atol = 1e-5
    reference_rtol = 1e-5

    identities = [
        (lambda x: sin_pi(x) ** 2 + cos_pi(x) ** 2, 1.0),  # Pythagorean identity
    ]
