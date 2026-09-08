import torch
import pytest
import mpmath as mp
import numpy as np
from constraints import int1, int2, nodes, f_delta, combined_weights1

mp.mp.dps = 30

Deltas = np.arange(2,11).tolist()

def f_delta_cross_check(Delta, x):
    f_cross_check = mp.hyp2f1(Delta + 1, Delta + 2, 2 * Delta +4, x)
    f_delta_cross_check_expr = (x**(Delta + 1)) / (1 - Delta) * f_cross_check

    return f_delta_cross_check_expr

def int1_cross_check(Delta):

    def integrand1(x):
        x = mp.mpf(x)
        t1 = (x - 1 - x ** 2) / (x ** 2)
        t2 = (1 - 2 * x) / (x * (1 - x))
        integrand1_cross_check_expr = -(t1 * t2 * f_delta_cross_check(Delta, x))
        return integrand1_cross_check_expr

    int1_cross_check_expr = float(mp.quad(integrand1, [0, 0.5]))

    return int1_cross_check_expr


def int2_cross_check(Delta):
    def integrand2(x):
        x = mp.mpf(x)
        t3 = (2 * x - 1) / (x ** 2)
        integrand2_cross_check_expr = t3 * f_delta_cross_check(Delta, x)
        return integrand2_cross_check_expr

    int2_cross_check_expr = float(mp.quad(integrand2, [0, 0.5]))

    return int2_cross_check_expr

def test_int1():
    for Delta_value in Deltas:
        Delta = torch.tensor(float(Delta_value), dtype=torch.float64)
        assert int1(Delta).item() == pytest.approx(int1_cross_check(Delta_value), rel=1e-6)

def test_int2():
    for Delta_value in Deltas:
        Delta = torch.tensor(float(Delta_value), dtype=torch.float64)
        assert int2(Delta).item() == pytest.approx(int2_cross_check(Delta_value), rel=1e-6)

def test_int1_gradient():
    for Delta_value in Deltas:
        Delta = torch.tensor(Delta_value, dtype=torch.float64, requires_grad=True)

        torch.autograd.gradcheck(int1, Delta)

def test_int2_gradient():
    for Delta_value in Deltas:
        Delta = torch.tensor(Delta_value, dtype=torch.float64, requires_grad=True)

        torch.autograd.gradcheck(int2, Delta)

def test_integrand1_asymptotic_value():
    Delta_test = torch.tensor([2.0] * len(nodes), dtype=torch.float64)
    f_evaluated = f_delta(Delta_test, nodes)
    integrand = f_evaluated * combined_weights1

    assert torch.all(torch.isfinite(integrand)) == True