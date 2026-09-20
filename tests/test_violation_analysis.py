import numpy as np
from violation_analysis import violation_by_state, ratio_to_truth
from bounds_sweep import violation_score
from spectrum_data import available_g_values
from ope_bounds_data import available_g_ope_values
from coupling import G_MIN
from convex_baseline_with_integrals import solve_convex_problem_with_integral_constraints
from spline_basis import build_knot_vector

test_g_values = [g for g in available_g_ope_values if g >= G_MIN]

g_values = available_g_values[1:]

knots = build_knot_vector(g_values[0], g_values[-1], 13, 5)
degree = 5
lam = 0.01

def test_violation_by_state_matches_violation_score():
    theta, _ = solve_convex_problem_with_integral_constraints(g_values, knots, degree, w_int = 1e6, lam = lam)

    total_from_breakdown = violation_by_state(theta, test_g_values, knots, degree)['violation'].sum()
    total_from_score = violation_score(theta, test_g_values, knots, degree)

    assert np.isclose(total_from_breakdown, total_from_score)

def test_ratio_to_truth_shape():
    theta, _ = solve_convex_problem_with_integral_constraints(g_values, knots, degree, w_int=1e6, lam=lam)

    for g in test_g_values:
        assert ratio_to_truth(theta, g, knots, degree).shape == (3, 4)

def test_violation_by_state_matches_violation_score2():
    theta, _ = solve_convex_problem_with_integral_constraints(g_values, knots, degree, w_int = 1e6, lam = lam)

    mixed_g_values = available_g_ope_values[::10]
    result = violation_by_state(theta, mixed_g_values, knots, degree)

    assert result['in_domain'].any()
    assert not result['in_domain'].all()