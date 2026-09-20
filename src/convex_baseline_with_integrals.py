import numpy as np
import cvxpy as cp
from stack_crossing import build_stacked_design_matrix, build_stacked_target_vector
from spline_basis import build_smoothness_penalty, build_knot_vector
from integral_constraints import build_stacked_integral_block, build_stacked_integral_target
from coupling import G_MIN
from spectrum_data import available_g_values

def solve_convex_problem_with_integral_constraints(g_values: list[float], knots: np.ndarray, degree: int, w_int: float, lam: float) -> tuple[np.ndarray, str]:
    r"""
        Solves the regularised non-negative B-spline regression problem with the integral constraints on top.

        Parameters
        ----------
        g_values: list[float]
            A list of values of the coupling constant, g, where the spline basis functions are evaluated.

        knots: np.ndarray
            A 1D array that contains the knot sequence including the endpoints.

        degree: int
            The polynomial degree of the B-spline function.

        w_int: float
            The regularisation hyperparameter for the integral constraint.

        lam: float
            The regularisation hyperparameter for the smoothness penalty.

        Returns
        -------
        theta: np.ndarray
             A 2D array of shape (10, n_basis) that contains the optimised non-negative B-spline coefficients.
        status: str
            The exit status string that is returned by the CVXPY solver.

        Notes
        -----
        The optimisation objective consists of:
        1. A crossing term that minimises data-driven regression residuals over all values of the coupling constant.
        2. An integral term that penalises deviations from expected physical constraints for large values of the coupling.
        3. A smoothness term that penalises the second-difference curvature of the spline profile.

        g_values: represents the entire training window. Intenrally, the function filters this down to int_g_values, namely
        those that satisfy g >= G_MIN. This happens for the integral term only, so the caller never needs to pass a
        pre-filtered list.

        Both cp.OPTIMAL and cp.OPTIMAL_INACCURATE states are explicitly permitted to accommodate numerical edge-cases
        common in tightly constrained regularisation paths. The status of the solver is returned along with the optimal
        parameters.

        w_int = 0.0 provides a direct mathematical reduction to the basic sltn_convex_problem;
        verified by the test__w_int_zero_matches_baseline
    """
    n_basis = len(knots) - degree - 1
    D = build_smoothness_penalty(n_basis)

    A_cross = build_stacked_design_matrix(g_values, knots, degree)
    b_cross = build_stacked_target_vector(g_values)

    int_g_values = [g for g in g_values if g >= G_MIN]

    A_int = build_stacked_integral_block(int_g_values, knots, degree)
    b_int = build_stacked_integral_target(int_g_values)

    theta = cp.Variable((10, n_basis))
    theta_flat = cp.reshape(theta, (10 * n_basis, ), order = 'C')

    crossing_term   = cp.sum_squares(A_cross @ theta_flat - b_cross)
    integral_term   = cp.sum_squares(A_int @ theta_flat - b_int)
    smoothness_term = cp.sum_squares(D @ theta.T)

    objective = cp.Minimize(crossing_term + w_int * integral_term + lam * smoothness_term)
    problem = cp.Problem(objective, [theta >= 0])
    problem.solve(solver = cp.CLARABEL)

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Solver status: {problem.status}")

    return theta.value, problem.status

if __name__ == '__main__':
    g_values = available_g_values[1:]
    knots = build_knot_vector(g_values[0], g_values[-1], 13, 5)
    theta, status = solve_convex_problem_with_integral_constraints(g_values, knots, 5, 0.0, 0.01)
    print(theta, status)
