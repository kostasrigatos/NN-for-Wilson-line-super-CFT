import numpy as np

from spectrum_data import available_g_values
from convex_baseline_with_integrals import solve_convex_problem_with_integral_constraints
from convex_baseline import solve_convex_problem
from spline_basis import build_knot_vector, build_smoothness_penalty
from stack_crossing import build_stacked_design_matrix, build_stacked_target_vector

g_values = available_g_values[1:]

knots = build_knot_vector(g_values[0], g_values[-1], 13, 5)
degree = 5
lam = 0.01
n_basis = len(knots) - degree - 1
D = build_smoothness_penalty(n_basis)

A_cross = build_stacked_design_matrix(g_values, knots, degree)
b_cross = build_stacked_target_vector(g_values)

def objective_value(theta, A_cross, b_cross, D, lam):
    resid = A_cross @ theta.flatten(order = 'C') - b_cross
    smooth = D @ theta.T
    result = np.sum(resid ** 2) + lam * np.sum(smooth ** 2)

    return result

def test__w_int_zero_matches_baseline():
    theta_baseline = solve_convex_problem(g_values, knots, degree, lam)
    theta_new, status = solve_convex_problem_with_integral_constraints(g_values, knots, degree, w_int = 0.0, lam = lam)

    obj_baseline = objective_value(theta_baseline, A_cross, b_cross, D, lam)
    obj_new      = objective_value(theta_new, A_cross, b_cross, D, lam)

    relative_difference = abs(obj_new - obj_baseline) / obj_baseline

    # Looser than a typical equivalence check: the two solvers agree on the
    # objective to ~1e-5, but land at slightly different points in the
    # sextet states (4-9), the same near-null-space directions flagged by
    # the identifiability analysis. Comparing objective values rather than
    # raw theta sidesteps that; see the discussion in this file's git history.

    assert relative_difference < 1e-4

def test_shape_and_nonnegativity_with_integrals():
    theta, status = solve_convex_problem_with_integral_constraints(g_values, knots, degree, w_int = 1e-6,lam=lam)
    assert theta.shape == (10, n_basis)
    assert np.all(theta >= -1e-4)

