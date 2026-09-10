import numpy as np
import math
import pytest
import mpmath as mp
import torch
from crossing import C2_BPS, f_I, f_B2, h
from reference import mu_D2_squared

gs = np.arange(5, 60, 5).tolist()
xs     = np.arange(0.5, 0.86, 0.05).tolist()

def test_f_I():
    for x_value in xs:
        assert f_I(x_value) == x_value

def test_asymptotic_C2_mu():
    for g_value in gs:
        assert C2_BPS(g_value) == pytest.approx(mu_D2_squared(1/(4*math.pi*g_value)))

def f_B2_cross_check(x):

    f2mpmath = x - x * mp.hyp2f1(1, 2, 4, x)
    return f2mpmath

def test_f_B2():
    for x_value in xs:
        assert f_B2_cross_check(x_value) == pytest.approx(f_B2(torch.tensor(float(x_value), dtype=torch.float64)))

def h_cross_check(g: float, x: torch.Tensor) -> torch.Tensor:
    term1 = (f_I(1-x) + C2_BPS(g)*f_B2(1-x))
    term2 = (f_I(x) + C2_BPS(g)*f_B2(x))
    h_expr = x**2 * term1 + (1 - x)**2 * term2

    return h_expr

def test_h():
    for g_value in gs:
        for x_value in xs:
            x = torch.tensor(x_value, dtype=torch.float64)
            assert h_cross_check(g_value, x) == pytest.approx(h(g_value, x), rel=1e-12)