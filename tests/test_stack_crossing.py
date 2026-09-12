import numpy as np

from design_matrix import build_design_matrix
from stack_crossing import build_theta_block
from spline_basis import build_knot_vector, build_basis_matrix
from spectrum_data import available_g_values

n_interior_knots_values = np.arange(1, 5, 1).tolist()
degree_values = np.arange(1, 10, 1).tolist()
g_min_val = available_g_values[1]
g_max_val = available_g_values[-1]

def test_build_theta_block1():
    for n_interior_knots_val in n_interior_knots_values:
        for degree_val in degree_values:
            knots = build_knot_vector(g_min_val, g_max_val, n_interior_knots_val, degree_val)
            g_values = available_g_values[1::10]
            for g_val in g_values:
                assert build_theta_block(g_val, knots, degree_val).shape == (350, 10 * (len(knots) - degree_val - 1))

def test_build_theta_block2():
    g_value = available_g_values[59]
    knots_value = build_knot_vector(g_min_val, g_max_val, 1, 5)
    n_basis = len(knots_value) - 5 - 1
    theta_random = np.random.rand(10, n_basis)
    resultA = build_theta_block(g_value, knots_value, 5) @ theta_random.flatten()

    stepB1 = build_basis_matrix(np.array([g_value]), knots_value, 5)[0]
    c = theta_random @ stepB1
    resultB = build_design_matrix(g_value) @ c

    assert np.allclose(resultA, resultB)