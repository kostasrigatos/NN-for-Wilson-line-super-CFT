import mpmath as mp
import pytest
import torch
import numpy as np
from blocks import hyper2f1, f_delta, F_delta

def hyper2f1_cross_check(Delta: float, x: float, n_terms: int = 300) -> float:

    a, b, c = Delta + 1, Delta + 2, 2 * Delta + 4

    hyper2f1_cross_check_expr = mp.mpf(0)

    for n in range(n_terms):
        pochhammer_symbols = mp.rf(a, n) * mp.rf(b, n) / mp.rf(c, n)
        x_factor = (mp.mpf(x)**n) / mp.factorial(n)
        hyper2f1_cross_check_expr += pochhammer_symbols * x_factor

    return hyper2f1_cross_check_expr

def test_hyper2f1():
    assert hyper2f1_cross_check(2, 1) == pytest.approx(hyper2f1(2, 1), rel=1e-3)
    for x in [0.5, 0.85]:
        for Delta in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
            assert hyper2f1_cross_check(Delta, x) == pytest.approx(hyper2f1(Delta, x), rel=1e-5)

xs     = np.arange(0.5, 0.86, 0.05).tolist()
Deltas = np.arange(2,11).tolist()

def test_F_delta_at_one():
    for Delta_value in Deltas:
        Delta = torch.tensor(float(Delta_value), dtype=torch.float64)
        x = torch.tensor(1.0, dtype=torch.float64)
        assert  F_delta(Delta, x).item() == pytest.approx(0.0, abs=1e-10)

def test_hyper2f1_gradient():
    for Delta_value in Deltas:
        for x_value in xs:
            Delta = torch.tensor(Delta_value, dtype=torch.float64, requires_grad=True)
            x = torch.tensor(x_value, dtype=torch.float64, requires_grad=True)

            torch.autograd.gradcheck(hyper2f1, (Delta, x))

def test_f_delta_gradient():
    for Delta_value in Deltas:
        for x_value in xs:
            Delta = torch.tensor(Delta_value, dtype=torch.float64, requires_grad=True)
            x = torch.tensor(x_value, dtype=torch.float64, requires_grad=True)

            torch.autograd.gradcheck(f_delta, (Delta, x))

def test_F_delta_gradient():
    for Delta_value in Deltas:
        for x_value in xs:
            Delta = torch.tensor(Delta_value, dtype=torch.float64, requires_grad=True)
            x = torch.tensor(x_value, dtype=torch.float64, requires_grad=True)

            torch.autograd.gradcheck(F_delta, (Delta, x))
