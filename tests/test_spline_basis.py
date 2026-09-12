from spline_basis import build_knot_vector, build_basis_matrix, build_smoothness_penalty
import numpy as np

g_min_values = np.arange(0.1, 0.3, 0.1).tolist()
g_max_values = np.arange(1.1, 1.3, 0.1).tolist()
n_interior_knots_values = np.arange(1, 20, 1).tolist()
degree_values = np.arange(1, 10, 1).tolist()


def test_build_knot_vector1():
    for g_min_val in g_min_values:
        for g_max_val in g_max_values:
            for n_interior_knots_val in n_interior_knots_values:
                for degree_val in degree_values:
                    assert build_knot_vector(g_min_val, g_max_val, n_interior_knots_val, degree_val).shape == (n_interior_knots_val + 2 * (degree_val + 1),)

def test_build_knot_vector2():
    for g_min_val in g_min_values:
        for g_max_val in g_max_values:
            for n_interior_knots_val in n_interior_knots_values:
                for degree_val in degree_values:
                    assert np.all(build_knot_vector(g_min_val, g_max_val, n_interior_knots_val, degree_val)[0: degree_val + 1] == np.full(degree_val + 1, g_min_val))

def test_build_knot_vector3():
    for g_min_val in g_min_values:
        for g_max_val in g_max_values:
            for n_interior_knots_val in n_interior_knots_values:
                for degree_val in degree_values:
                    assert np.all(build_knot_vector(g_min_val, g_max_val, n_interior_knots_val, degree_val) >= 0)

def test_build_knot_vector4():
    for g_min_val in g_min_values:
        for g_max_val in g_max_values:
            for n_interior_knots_val in n_interior_knots_values:
                for degree_val in degree_values:
                    assert np.all(build_knot_vector(g_min_val, g_max_val, n_interior_knots_val, degree_val)[-(degree_val + 1):] == np.full(degree_val + 1, g_max_val))

def test_build_knot_vector5():
    for g_min_val in g_min_values:
        for g_max_val in g_max_values:
            for n_interior_knots_val in n_interior_knots_values:
                for degree_val in degree_values:
                    assert np.all(np.diff(build_knot_vector(g_min_val, g_max_val, n_interior_knots_val, degree_val)) >= 0 )

def test_build_basis_matrix1():
    for g_min_val in g_min_values:
        for g_max_val in g_max_values:
            for n_interior_knots_val in n_interior_knots_values:
                for degree_val in degree_values:
                    knots = build_knot_vector(g_min_val, g_max_val, n_interior_knots_val, degree_val)
                    g_values = np.linspace(g_min_val, g_max_val, 10)
                    assert build_basis_matrix(g_values, knots, degree_val).shape == (len(g_values), len(knots) - degree_val - 1)

def test_build_basis_matrix2():
    for g_min_val in g_min_values:
        for g_max_val in g_max_values:
            for n_interior_knots_val in n_interior_knots_values:
                for degree_val in degree_values:
                    knots = build_knot_vector(g_min_val, g_max_val, n_interior_knots_val, degree_val)
                    g_values = np.linspace(g_min_val, g_max_val, 10)
                    row_sums = np.sum(build_basis_matrix(g_values, knots, degree_val), axis=1)
                    assert np.allclose(row_sums, 1.0)

def test_build_basis_matrix3():
    for g_min_val in g_min_values:
        for g_max_val in g_max_values:
            for n_interior_knots_val in n_interior_knots_values:
                for degree_val in degree_values:
                    knots = build_knot_vector(g_min_val, g_max_val, n_interior_knots_val, degree_val)
                    g_values = np.linspace(g_min_val, g_max_val, 10)
                    assert np.all(build_basis_matrix(g_values, knots, degree_val) >= 0)

def expected_second_diff(n):
    expected = np.zeros((n - 2, n))
    for k in range(n - 2):
        expected[k, k] = 1
        expected[k, k+1] = -2
        expected[k, k+2] = 1
    return expected

n_values = list(range(3,20))
def test_build_smoothness_penalty():
    for n_number in n_values:
        assert np.array_equal(build_smoothness_penalty(n_number),expected_second_diff(n_number))
