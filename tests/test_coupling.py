import pytest
import math
import mpmath as mp
from coupling import B_function, F_function, F_function_cross_check

def B_function_cross_check(g, dps = 50):
    mp.mp.dps = dps
    B_function_cross_check_expr = (g / math.pi) * (mp.besseli(2, 4 * math.pi * g)) / (mp.besseli(1, 4 * math.pi * g))
    return B_function_cross_check_expr

def test_B_function():
    for g in [0.1, 1, 5, 10, 50, 100, 500]:
        assert B_function(g) == pytest.approx(B_function_cross_check(g), rel=1e-12)

def test_F_function():
    for g in [0.1, 1, 5, 10, 50, 100, 500]:
        assert F_function(g) == pytest.approx(F_function_cross_check(g), rel=1e-12)