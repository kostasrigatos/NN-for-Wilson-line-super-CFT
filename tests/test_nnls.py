import numpy as np
import pytest
from nnls import sltn_nnls
from design_matrix import build_target_vector, build_design_matrix

gs_real = np.arange(0.1, 3, 0.1).tolist()

def test_nnls1():
    for g_value in gs_real:
        assert sltn_nnls(g_value)[0].shape == (10,)


def test_nnls2():
    for g_value in gs_real:
        assert type(sltn_nnls(g_value)[1]) == float

def test_nnls3():
    for g_value in gs_real:
        assert np.all(sltn_nnls(g_value)[0] >= 0)

def test_nnls4():
    for g_value in gs_real:
        assert np.all(np.isfinite(sltn_nnls(g_value)[0]))

def test_nnls5():
    for g_value in gs_real:
        assert np.all(np.isfinite(sltn_nnls(g_value)[1]))

def test_nnls6():
    for g_value in gs_real:
        A = build_design_matrix(g_value)
        b = build_target_vector(g_value)
        sltn_vector, residual_norm = sltn_nnls(g_value)
        assert np.linalg.norm(A@sltn_vector - b) == pytest.approx(residual_norm)