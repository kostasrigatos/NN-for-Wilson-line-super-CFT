import numpy as np
from design_matrix import build_target_vector, z_points, z_tensor, build_design_matrix
from crossing import h

gs = np.arange(5, 60, 5).tolist()
gs_cv = np.arange(65, 120, 5).tolist()
gs_real = np.arange(0, 4, 0.1).tolist()

def test_build_target_vector_1():
    for g_value in gs:
        b = build_target_vector(g_value)
        assert len(b) == len(z_points)
        assert b.dtype == np.float64
        assert isinstance(b, np.ndarray)

def test_build_target_vector_2():
    for g_value in gs:
        assert np.allclose(build_target_vector(g_value), -h(g_value, z_tensor).numpy())

def test_build_target_vector_3():
    for g_value, g_cv_value in zip(gs, gs_cv):
        assert (build_target_vector(g_value) != build_target_vector(g_cv_value)).any()

def test_build_design_matrix_1():
    for g_value in gs_real:
        assert build_design_matrix(g_value).shape == (350, 10)
