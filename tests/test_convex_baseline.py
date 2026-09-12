import numpy as np
from convex_baseline import solve_convex_problem, reconstruct_C_squared
from spline_basis import build_knot_vector, build_smoothness_penalty
from spectrum_data import available_g_values

g_values_test = available_g_values[1:15:3]
g_min_val, g_max_val = g_values_test[0], g_values_test[-1]
knots_test = build_knot_vector(g_min_val, g_max_val, 4, 3)
degree_test = 3
n_basis_test = len(knots_test) - degree_test - 1

def test_solve_convex_problem_shape():
    theta = solve_convex_problem(g_values_test, knots_test, degree_test, lam=0.01)
    assert theta.shape == (10, n_basis_test)

def test_solve_convex_problem_nonnegativity():
    theta = solve_convex_problem(g_values_test, knots_test, degree_test, lam=0.01)
    assert np.all(theta >= -1e-4)

def test_solve_convex_problem_lam_increases_smoothness():
    D = build_smoothness_penalty(n_basis_test)
    theta_low_lam = solve_convex_problem(g_values_test, knots_test, degree_test, lam=0.001)
    theta_high_lam = solve_convex_problem(g_values_test, knots_test, degree_test, lam=10.0)
    penalty_low = np.sum((D @ theta_low_lam.T) ** 2)
    penalty_high = np.sum((D @ theta_high_lam.T) ** 2)
    assert penalty_high < penalty_low

def test_reconstruct_C_squared_shape():
    theta = solve_convex_problem(g_values_test, knots_test, degree_test, lam=0.01)
    c_squared = reconstruct_C_squared(theta, g_values_test[2], knots_test, degree_test)
    assert c_squared.shape == (10,)

def test_reconstruct_C_squared_nonnegativity():
    theta = solve_convex_problem(g_values_test, knots_test, degree_test, lam=0.01)
    c_squared = reconstruct_C_squared(theta, g_values_test[2], knots_test, degree_test)
    assert np.all(c_squared >= -1e-4)