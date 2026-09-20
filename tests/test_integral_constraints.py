import numpy as np
import pytest
from spectrum_data import available_g_values
from coupling import G_MIN
from integral_constraints import build_integral_design, build_integral_target, build_integral_block, build_stacked_integral_block, build_stacked_integral_target, _check_g_values_valid
from spline_basis import build_knot_vector, build_basis_matrix

g_values_test = [g for g in available_g_values if g >= G_MIN]
degree_values = np.arange(1, 10, 1).tolist()
n_interior_knots_values = np.arange(1, 5, 1).tolist()
g_min_val = available_g_values[1]
g_max_val = available_g_values[-1]

def test_build_integral_design_shape():
    for g in g_values_test:
        assert build_integral_design(g).shape == (2,10)

def test_regression_values():
    assert np.all(np.isclose(build_integral_design(3.0)[:,0], [-0.526977, 0.087281]))
    assert np.all(np.isclose(build_integral_target(3.0), [-0.19385, 0.032381], atol=1e-5))

def test_build_integral_block_shape():
    for n_interior_knots_val in n_interior_knots_values:
        for degree_val in degree_values:
            knots = build_knot_vector(g_min_val, g_max_val, n_interior_knots_val, degree_val)
            g_values = available_g_values[1::10]
            for g_val in g_values:
                assert build_integral_block(g_val, knots, degree_val).shape == (2, 10 * (len(knots) - degree_val - 1))

def test_two_paths_cross():
    g_value = available_g_values[59]
    knots_value = build_knot_vector(g_min_val, g_max_val, 1, 5)
    n_basis = len(knots_value) - 5 - 1
    theta_random = np.random.rand(10, n_basis)
    resultA = build_integral_block(g_value, knots_value, 5) @ theta_random.flatten()

    basis_row = build_basis_matrix(np.array([g_value]), knots_value, 5)[0]
    c = theta_random @ basis_row
    resultB = build_integral_design(g_value) @ c

    assert np.allclose(resultA, resultB)

def test_stacked_shapes():
    for n_interior_knots_val in n_interior_knots_values:
        for degree_val in degree_values:
            knots = build_knot_vector(g_min_val, g_max_val, n_interior_knots_val, degree_val)
            g_values = g_values_test
            assert build_stacked_integral_block(g_values, knots, degree_val).shape == (2 * len(g_values), 10 * (len(knots) - degree_val - 1))
            assert build_stacked_integral_target(g_values).shape == (2 * len(g_values), )

def test_raise_error():
    with pytest.raises(ValueError):
        _check_g_values_valid(available_g_values)
